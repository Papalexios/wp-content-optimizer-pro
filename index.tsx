
import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import React, { useState, useMemo, useEffect, useCallback, useReducer, useRef } from 'react';
import ReactDOM from 'react-dom/client';

// Debounce function to limit how often a function gets called
const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
};

/**
 * Returns a random subset of an array using the Fisher-Yates shuffle algorithm.
 * @param array The source array.
 * @param size The size of the random subset to return.
 * @returns A new array containing the random subset.
 */
const getRandomSubset = (array, size) => {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled.slice(0, size);
};


/**
 * Extracts a JSON object from a string that might be wrapped in markdown,
 * have leading/trailing text, or other common AI response artifacts.
 * @param text The raw string response from the AI.
 * @returns The clean JSON string.
 * @throws {Error} if a valid JSON object cannot be found.
 */
const extractJson = (text: string): string => {
    // 1. Look for a JSON markdown block
    const markdownMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (markdownMatch && markdownMatch[1]) {
        try {
            JSON.parse(markdownMatch[1].trim());
            return markdownMatch[1].trim();
        } catch (e) {
            // Fall through if markdown content is not valid JSON
        }
    }

    // 2. If no markdown, find the first '{' or '[' and last '}' or ']'
    const firstBracket = text.indexOf('{');
    const lastBracket = text.lastIndexOf('}');
    const firstSquare = text.indexOf('[');
    const lastSquare = text.lastIndexOf(']');

    let start = -1;
    let end = -1;

    if (firstBracket !== -1 && lastBracket > firstBracket) {
        start = firstBracket;
        end = lastBracket;
    }
    
    // Check if a square bracket JSON array is more likely
    if (firstSquare !== -1 && lastSquare > firstSquare && (start === -1 || firstSquare < start)) {
        start = firstSquare;
        end = lastSquare;
    }


    if (start !== -1 && end > start) {
        return text.substring(start, end + 1);
    }

    throw new Error("Could not find a valid JSON object in the AI response.");
};


/**
 * A professional, state-of-the-art promise queue processor.
 * It executes a series of promise-returning functions sequentially with a delay,
 * preventing API rate-limiting issues and providing progress updates.
 * @param items The array of items to process.
 * @param promiseFn The function that takes an item and returns a promise.
 * @param onProgress Optional callback to report progress for each item.
 * @param delay The delay in ms between each promise execution.
 */
const processPromiseQueue = async (items, promiseFn, onProgress, delay = 1000) => {
    const results = [];
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        try {
            const result = await promiseFn(item);
            results.push({ status: 'fulfilled', value: result });
            if (onProgress) onProgress({ item, result, index: i, success: true });
        } catch (error) {
            results.push({ status: 'rejected', reason: error });
            if (onProgress) onProgress({ item, error, index: i, success: false });
        }
        if (i < items.length - 1) {
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    return results;
};

/**
 * Wraps an async function with a robust retry mechanism. It retries on any failure,
 * using exponential backoff for API rate-limiting errors (HTTP 429) and a short,
 * constant delay for other transient errors (e.g., network issues, JSON parsing).
 * @param apiCallFn The async function to call, which should handle the entire process
 * including parsing and validation, throwing an error on failure.
 * @param maxRetries The maximum number of retries before giving up.
 * @param initialDelay The initial delay in ms for the first rate-limit retry.
 * @returns A Promise that resolves with the result of the `apiCallFn`.
 */
const makeResilientAiCall = async <T,>(
    apiCallFn: () => Promise<T>, 
    maxRetries: number = 3, 
    initialDelay: number = 2000
): Promise<T> => {
    let lastError: Error | null = null;
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await apiCallFn();
        } catch (error: any) {
            lastError = error;
            if (i >= maxRetries - 1) {
                console.error(`AI call failed on final attempt (${maxRetries}).`, error);
                throw error; // Throw after final attempt
            }

            const isRateLimitError = (
                (error.status === 429) || 
                (error.message && error.message.includes('429')) ||
                (error.message && error.message.toLowerCase().includes('rate limit'))
            );

            let delay = 1000; // Default delay for non-rate-limit errors
            if (isRateLimitError) {
                delay = initialDelay * Math.pow(2, i) + Math.random() * 1000;
                console.warn(`Rate limit error detected. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${i + 1}/${maxRetries})`);
            } else {
                 console.warn(`AI call failed. Retrying in ${delay/1000}s... (Attempt ${i + 1}/${maxRetries})`, error.message);
            }
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    // This part should be unreachable if maxRetries > 0, but is a good fallback.
    throw new Error(`AI call failed after ${maxRetries} attempts. Last error: ${lastError?.message}`);
};


/**
 * Intelligently fetches a public resource (e.g., sitemap) by first attempting a direct connection.
 * If the direct connection fails due to a CORS-like network error, it automatically
 * falls back to a series of reliable CORS proxies. **Primarily intended for public GET requests.**
 * @param url The target URL to fetch.
 * @param options The standard fetch options object.
 * @returns A Promise that resolves with the Response object.
 */
const smartFetch = async (url: string, options: RequestInit = {}): Promise<Response> => {
    try {
        const directResponse = await fetch(url, options);
        if (directResponse.ok) return directResponse;
        console.warn(`Direct fetch to ${url} was not OK, status: ${directResponse.status}. Trying proxies.`);
    } catch (error) {
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
            console.warn(`Direct fetch to ${url} failed, likely due to CORS. Falling back to proxies.`);
        } else {
            console.error('An unexpected network error occurred during direct fetch:', error);
            throw error;
        }
    }

    const proxies = [
        { name: 'corsproxy.io', buildUrl: (targetUrl) => `https://corsproxy.io/?${targetUrl}`},
        { name: 'allorigins.win', buildUrl: (targetUrl) => `https://api.allorigins.win/raw?url=${encodeURIComponent(targetUrl)}`},
        { name: 'thingproxy.freeboard.io', buildUrl: (targetUrl) => `https://thingproxy.freeboard.io/fetch/${targetUrl}`},
        { name: 'CodeTabs', buildUrl: (targetUrl) => `https://api.codetabs.com/v1/proxy?quest=${targetUrl}`}
    ];
    let lastError: Error | null = new Error('No proxies were attempted.');

    for (const proxy of proxies) {
        try {
            const proxyResponse = await fetch(proxy.buildUrl(url), options);
            if (proxyResponse.ok) {
                console.log(`Successfully fetched via proxy: ${proxy.name}`);
                return proxyResponse;
            }
            lastError = new Error(`Proxy ${proxy.name} returned status ${proxyResponse.status}`);
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));
        }
    }
    console.error("All proxies failed.", lastError);
    throw new Error(`All proxies failed to fetch the resource. Last error: ${lastError.message}`);
};

/**
 * Performs a direct fetch request without proxy fallbacks.
 * Provides a more informative error message for common CORS issues,
 * which is crucial for authenticated API calls where proxies are not suitable.
 * @param url The target URL to fetch.
 * @param options The standard fetch options object.
 * @returns A Promise that resolves with the Response object.
 */
const directFetch = async (url: string, options: RequestInit = {}): Promise<Response> => {
    try {
        const response = await fetch(url, options);
        return response;
    } catch (error) {
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
            throw new Error(
                "A network error occurred, likely due to a CORS policy on your server. " +
                "Please ensure your WordPress URL is correct and that your server is configured to accept requests. " +
                "Using a browser extension to disable CORS can be a temporary workaround for development."
            );
        }
        throw error;
    }
};

/**
 * Recursively parses a sitemap or sitemap index to extract all unique URLs.
 * @param url The URL of the sitemap or sitemap index.
 * @param visited A Set to keep track of visited sitemap URLs to prevent infinite loops.
 * @returns A Promise resolving to an array of all found URLs.
 */
const parseSitemap = async (url: string, visited: Set<string> = new Set()): Promise<string[]> => {
    if (visited.has(url)) return [];
    visited.add(url);

    try {
        const response = await smartFetch(url);
        if (!response.ok) {
            console.error(`Failed to fetch sitemap/index at ${url}. Status: ${response.status}`);
            return []; // Fail gracefully for this URL
        }
        
        const text = await response.text();
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(text, "application/xml");

        if (xmlDoc.getElementsByTagName("parsererror").length > 0) {
            console.error(`XML parsing error for ${url}`);
            return [];
        }

        const isSitemapIndex = xmlDoc.getElementsByTagName('sitemapindex').length > 0;
        if (isSitemapIndex) {
            const sitemapUrls = Array.from(xmlDoc.getElementsByTagName('loc')).map(node => node.textContent).filter(Boolean) as string[];
            const promises = sitemapUrls.map(sitemapUrl => parseSitemap(sitemapUrl, visited));
            const results = await Promise.all(promises);
            return results.flat();
        }

        const isUrlset = xmlDoc.getElementsByTagName('urlset').length > 0;
        if (isUrlset) {
            return Array.from(xmlDoc.getElementsByTagName('loc')).map(node => node.textContent).filter(Boolean) as string[];
        }
        
        console.warn(`No <sitemapindex> or <urlset> found in ${url}.`);
        return [];

    } catch (error) {
        console.error(`Error processing sitemap URL ${url}:`, error);
        return [];
    }
};

const slugToTitle = (url: string): string => {
    try {
        const path = new URL(url).pathname;
        if (path === '/blog/') return 'Affiliate Marketing Blog';
        return path.replace(/^\/|\/$/g, '').split('/').pop().replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    } catch (e) {
        return url;
    }
};

const ProgressBar = ({ currentStep }: { currentStep: number }) => {
    const steps = ['Config', 'Manage Content', 'Review & Publish'];
    return (
        <ol className="progress-bar">
            {steps.map((name, index) => {
                const stepIndex = index + 1;
                const status = stepIndex < currentStep ? 'completed' : stepIndex === currentStep ? 'active' : '';
                return (
                    <li key={name} className={`progress-step ${status}`}>
                        <div className="step-circle">{stepIndex < currentStep ? '✔' : stepIndex}</div>
                        <span className="step-name">{name}</span>
                    </li>
                );
            })}
        </ol>
    );
};

const ApiKeyValidator = ({ status }) => {
    if (status === 'validating') return <div className="key-status-icon"><div className="key-status-spinner"></div></div>;
    if (status === 'valid') return <div className="key-status-icon success"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg></div>;
    if (status === 'invalid') return <div className="key-status-icon error"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg></div>;
    return null;
};

const Feature = ({ icon, title, children }) => (
    <div className="feature">
        <div className="feature-icon">{icon}</div>
        <div className="feature-content">
            <h3>{title}</h3>
            <p>{children}</p>
        </div>
    </div>
);

const PromotionalLinks = () => {
    const linksToShow = useMemo(() => getRandomSubset(PROMOTIONAL_LINKS, 4), []);

    return (
        <div className="promo-links-section">
            <h3>Explore Our Expertise</h3>
            <p>Check out some of our top-performing content at affiliatemarketingforsuccess.com.</p>
            <div className="promo-links-grid">
                {linksToShow.map(url => (
                    <a key={url} href={url} target="_blank" rel="noopener noreferrer" className="promo-link-card">
                        <h4>{slugToTitle(url)}</h4>
                        <span>affiliatemarketingforsuccess.com</span>
                    </a>
                ))}
            </div>
        </div>
    );
};

const LandingPageIntro = () => (
    <div className="landing-intro">
        <h2 className="usp-headline">The intelligent engine that elevates your content from good to #1.</h2>
        <p className="usp-subheadline">
            Go beyond generic AI writing. We analyze your content landscape to create strategically-focused articles, 10x better than the competition.
        </p>
        <div className="features-grid">
            <Feature
                icon={<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13V7m0 13a2 2 0 002-2V9a2 2 0 00-2-2m-2 4h.01M15 20l5.447-2.724A1 1 0 0021 16.382V5.618a1 1 0 00-1.447-.894L15 7m0 13V7m0 13a2 2 0 01-2-2V9a2 2 0 012-2m2 4h-.01" /></svg>}
                title="Strategic Sitemap Analysis"
            >
                Unlike other tools, we analyze your sitemap to find high-value content gaps and expand your topical authority.
            </Feature>
            <Feature
                icon={<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>}
                title="10x Content Generation"
            >
                Produce comprehensive, SEO-optimized, and internally-linked articles with a single click, ready to rank.
            </Feature>
            <Feature
                icon={<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
                title="Existing Content Supercharger"
            >
                Revitalize your old posts by completely rewriting them into definitive, up-to-date, and authoritative resources.
            </Feature>
        </div>
        <PromotionalLinks />
        <div className="risk-reversal">
            <p><strong>Your Advantage:</strong> This is a powerful, free tool designed to give you a competitive edge. There are no trials or fees. Simply configure your details below and start optimizing.</p>
        </div>
    </div>
);


const ConfigStep = ({ state, dispatch, onFetchSitemap, onValidateKey }) => {
    const { wpUrl, wpUser, wpPassword, sitemapUrl, urlLimit, loading, aiProvider, apiKeys, openRouterModels, keyStatus } = state;
    const isSitemapConfigValid = useMemo(() => sitemapUrl && sitemapUrl.trim() !== '', [sitemapUrl]);
    const isApiKeyValid = useMemo(() => apiKeys[aiProvider]?.trim() && keyStatus[aiProvider] !== 'invalid', [apiKeys, aiProvider, keyStatus]);
    const [saveConfig, setSaveConfig] = useState(true);

    const debouncedValidateKey = useCallback(debounce(onValidateKey, 500), [onValidateKey]);

    const handleApiKeyChange = (e) => {
        const { value } = e.target;
        dispatch({ type: 'SET_API_KEY', payload: { provider: aiProvider, key: value } });
        if (value.trim() !== '') debouncedValidateKey(aiProvider, value);
    };

    const handleProviderChange = (e) => {
        const newProvider = e.target.value;
        dispatch({ type: 'SET_AI_PROVIDER', payload: newProvider });
        const key = apiKeys[newProvider];
        if (key?.trim() && keyStatus[newProvider] === 'unknown') onValidateKey(newProvider, key);
    };

    return (
        <div className="step-container">
            <LandingPageIntro />
            <div className="config-forms-wrapper">
                <fieldset className="config-fieldset">
                    <legend>WordPress Configuration</legend>
                    <div className="form-group"><label htmlFor="wpUrl">WordPress URL</label><input type="url" id="wpUrl" value={wpUrl} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpUrl', value: e.target.value } })} placeholder="https://example.com" /></div>
                    <div className="form-group"><label htmlFor="wpUser">WordPress Username</label><input type="text" id="wpUser" value={wpUser} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpUser', value: e.target.value } })} placeholder="admin" /></div>
                    <div className="form-group"><label htmlFor="wpPassword">Application Password</label><input type="password" id="wpPassword" value={wpPassword} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpPassword', value: e.target.value } })} placeholder="••••••••••••••••" /><p className="help-text">This is not your main password. <a href="https://wordpress.org/documentation/article/application-passwords/" target="_blank" rel="noopener noreferrer">Learn how to create one</a>.</p></div>
                    <div className="checkbox-group"><input type="checkbox" id="saveConfig" checked={saveConfig} onChange={(e) => setSaveConfig(e.target.checked)} /><label htmlFor="saveConfig">Save WordPress Configuration</label></div>
                    <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: '8px', backgroundColor: 'var(--warning-bg-color)', border: '1px solid var(--warning-color)', color: 'var(--warning-text-color)' }}><p style={{margin: 0, fontSize: '0.875rem', lineHeight: '1.5'}}><strong>Security Note:</strong> This app connects directly to your WordPress site. Connection issues (CORS) may require server configuration. Always use a dedicated Application Password with limited permissions.</p></div>
                </fieldset>

                <fieldset className="config-fieldset">
                    <legend>Content Source</legend>
                    <div className="form-group"><label htmlFor="sitemapUrl">Sitemap URL (or Sitemap Index URL)</label><input type="url" id="sitemapUrl" value={sitemapUrl} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'sitemapUrl', value: e.target.value } })} placeholder="https://example.com/sitemap.xml" /></div>
                </fieldset>

                <fieldset className="config-fieldset">
                    <legend>AI Configuration</legend>
                    <div className="form-group"><label htmlFor="aiProvider">AI Provider</label><select id="aiProvider" value={aiProvider} onChange={handleProviderChange}><option value="gemini">Google Gemini</option><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="openrouter">OpenRouter (Experimental)</option></select></div>
                    {aiProvider === 'openrouter' && (<div className="form-group"><label htmlFor="openRouterModel">Model</label><input type="text" id="openRouterModel" list="openrouter-models-list" value={state.openRouterModel} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'openRouterModel', value: e.target.value } })} placeholder="e.g., google/gemini-flash-1.5" /><datalist id="openrouter-models-list">{openRouterModels.map(model => <option key={model} value={model} />)}</datalist><p className="help-text">Enter any model name from <a href="https://openrouter.ai/models" target="_blank" rel="noopener noreferrer">OpenRouter</a>.</p></div>)}
                    <div className="form-group api-key-group"><label htmlFor="apiKey">API Key</label><input type="password" id="apiKey" value={apiKeys[aiProvider] || ''} onChange={handleApiKeyChange} placeholder={`Enter your ${aiProvider.charAt(0).toUpperCase() + aiProvider.slice(1)} API Key`} /><ApiKeyValidator status={keyStatus[aiProvider]} /></div>
                </fieldset>
            </div>

            <button className="btn" onClick={() => onFetchSitemap(sitemapUrl, saveConfig)} disabled={loading || !isSitemapConfigValid || !isApiKeyValid}>{loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Fetch Sitemap & Continue'}</button>
        </div>
    );
};

const NewContentHub = ({ onGenerate, isGenerating, onGenerateIdeas, isGeneratingTopics, suggestedTopics }) => {
    const [topic, setTopic] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (topic.trim() && !isGenerating) {
            onGenerate(topic);
        }
    };

    return (
        <div className="new-content-hub">
            <div className="topic-suggester">
                <h2>AI Content Strategist</h2>
                <p>Let our AI analyze your site and suggest high-impact pillar posts to build your topical authority and boost organic traffic.</p>
                <button className="btn" onClick={onGenerateIdeas} disabled={isGeneratingTopics}>
                    {isGeneratingTopics ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Generate SEO Topic Ideas'}
                </button>
                {suggestedTopics.length > 0 && (
                    <div className="suggestions-list">
                        {suggestedTopics.map((idea, index) => (
                            <div className="suggestion-card" key={index}>
                                <h4>{idea.title}</h4>
                                <p>{idea.description}</p>
                                <button className="btn btn-secondary" onClick={() => onGenerate(idea.title)} disabled={isGenerating}>
                                    {isGenerating ? 'Busy...' : 'Write This Article'}
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
            
            <div className="manual-topic-entry">
                <h2>Or, Create Your Own Topic</h2>
                <p>Enter your target keyword or a full blog post title below. The AI will generate a comprehensive, 1800+ word article designed to rank #1.</p>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="newTopic">Topic or Keyword</label>
                        <input
                            type="text"
                            id="newTopic"
                            value={topic}
                            onChange={(e) => setTopic(e.target.value)}
                            placeholder="e.g., How to start affiliate marketing in 2025"
                            disabled={isGenerating}
                        />
                    </div>
                    <button type="submit" className="btn" disabled={!topic.trim() || isGenerating}>
                        {isGenerating ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Generate Article'}
                    </button>
                </form>
            </div>
        </div>
    );
};

const ExistingContentTable = ({ state, dispatch, onGenerateContent, onGenerateAll, onFetchExistingPosts }) => {
    const { posts, loading, generationStatus, selectedPostIds, searchTerm, sortConfig, bulkGenerationProgress } = state;
    
    const filteredPosts = useMemo(() => {
        if (!searchTerm) return posts;
        return posts.filter(post => 
            post.title.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }, [posts, searchTerm]);

    const sortedPosts = useMemo(() => {
        const sorted = [...filteredPosts];
        if (sortConfig.key) {
            sorted.sort((a, b) => {
                const aVal = a[sortConfig.key];
                const bVal = b[sortConfig.key];
                if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
                if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
                return 0;
            });
        }
        return sorted;
    }, [filteredPosts, sortConfig]);
    
    const handleSort = (key) => {
        const direction = sortConfig.key === key && sortConfig.direction === 'asc' ? 'desc' : 'asc';
        dispatch({ type: 'SET_SORT_CONFIG', payload: { key, direction } });
    };
    
    const handleSelectAll = () => {
        const allVisibleIds = sortedPosts.map(p => p.id);
        dispatch({ type: 'SELECT_ALL_VISIBLE', payload: allVisibleIds });
    };

    const allVisibleSelected = sortedPosts.length > 0 && sortedPosts.every(p => selectedPostIds.has(p.id));
    
    const generatableCount = useMemo(() => {
        return [...selectedPostIds].filter(id => {
            const status = generationStatus[String(id)];
            return status !== 'done' && status !== 'generating';
        }).length;
    }, [selectedPostIds, generationStatus]);

    const isGenerateAllDisabled = bulkGenerationProgress.visible || generatableCount === 0;

    return (
         <div className="step-container full-width">
            {posts.length === 0 && !loading ? (
                <div className="fetch-posts-prompt">
                    <p>Ready to update your existing content?</p>
                    <button className="btn" onClick={onFetchExistingPosts} disabled={loading}>
                        {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Fetch Recent Posts'}
                    </button>
                </div>
            ) : (
                <>
                    <div className="table-toolbar">
                        <input
                           type="search"
                           className="table-search-input"
                           placeholder="Search posts by title..."
                           value={searchTerm}
                           onChange={e => dispatch({ type: 'SET_SEARCH_TERM', payload: e.target.value })}
                        />
                        <div className="selection-toolbar-actions">
                            {selectedPostIds.size > 0 && (
                                <>
                                    <span>{selectedPostIds.size} post{selectedPostIds.size !== 1 ? 's' : ''} selected</span>
                                    <button className="btn btn-secondary btn-small" onClick={() => dispatch({ type: 'DESELECT_ALL' })}>Deselect All</button>
                                </>
                            )}
                            <button className="btn btn-small" onClick={onGenerateAll} disabled={isGenerateAllDisabled}>
                                {bulkGenerationProgress.visible ? 'Generating...' : `Generate for ${generatableCount} Selected`}
                            </button>
                        </div>
                    </div>
                    
                    {bulkGenerationProgress.visible && (
                        <div className="bulk-progress-bar">
                            <div 
                                className="bulk-progress-bar-fill" 
                                style={{ width: `${(bulkGenerationProgress.current / bulkGenerationProgress.total) * 100}%` }}
                            ></div>
                            <span className="bulk-progress-bar-text">
                                Generating {bulkGenerationProgress.current} of {bulkGenerationProgress.total} posts...
                            </span>
                        </div>
                    )}

                    <div className="table-container">
                        <table className="content-table mobile-cards">
                            <thead>
                                <tr>
                                    <th className="checkbox-cell"><input type="checkbox" onChange={handleSelectAll} checked={allVisibleSelected} /></th>
                                    <th className="sortable" onClick={() => handleSort('title')}>
                                        Title
                                        {sortConfig.key === 'title' && <span className={`sort-icon ${sortConfig.direction}`}></span>}
                                    </th>
                                    <th className="sortable" onClick={() => handleSort('modified')}>
                                        Last Updated
                                        {sortConfig.key === 'modified' && <span className={`sort-icon ${sortConfig.direction}`}></span>}
                                    </th>
                                    <th>Status</th>
                                    <th className="actions-cell">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {loading && posts.length === 0 ? (
                                    <tr><td colSpan="5" style={{textAlign: 'center', padding: '2rem'}}><div className="spinner" style={{width: '32px', height: '32px', margin: '0 auto'}}></div></td></tr>
                                ) : sortedPosts.map(post => {
                                    const status = generationStatus[String(post.id)] || 'idle';
                                    const isSelected = selectedPostIds.has(post.id);
                                    return (
                                        <tr key={post.id} className={`${isSelected ? 'selected' : ''} status-row-${status}`}>
                                            <td className="checkbox-cell">
                                                <input
                                                    type="checkbox"
                                                    checked={isSelected}
                                                    onChange={() => dispatch({ type: 'TOGGLE_POST_SELECTION', payload: post.id })}
                                                />
                                            </td>
                                            <td data-label="Title"><a href={post.url} target="_blank" rel="noopener noreferrer">{post.title}</a></td>
                                            <td data-label="Last Updated">{new Date(post.modified).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</td>
                                            <td data-label="Status">
                                                <div className={`status status-${status}`}>
                                                    <span className="status-dot"></span>
                                                    {status === 'generating' ? 'Generating...' : status === 'done' ? 'Generated' : status === 'error' ? 'Error' : 'Ready to Update'}
                                                </div>
                                            </td>
                                            <td data-label="Actions" className="actions-cell">
                                                {status !== 'done' && (
                                                    <button className="btn btn-secondary btn-small" onClick={() => onGenerateContent(post)} disabled={status === 'generating' || bulkGenerationProgress.visible}>
                                                        {status === 'generating' ? <div className="spinner" style={{width: '18px', height: '18px'}}></div> : 'Generate'}
                                                    </button>
                                                )}
                                                {status === 'done' && (
                                                    <button className="btn btn-small" onClick={() => dispatch({ type: 'OPEN_REVIEW_MODAL', payload: posts.findIndex(p => String(p.id) === String(post.id)) })}>
                                                        Review
                                                    </button>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
         </div>
    );
};

const ContentStep = ({ state, dispatch, onGenerateContent, onFetchExistingPosts, onGenerateAll, onGenerateTopicIdeas }) => {
    const { contentMode, loading, isGeneratingTopics, suggestedTopics } = state;

    return (
        <div className="step-container">
            <div className="content-mode-toggle">
                <button className={contentMode === 'new' ? 'active' : ''} onClick={() => dispatch({ type: 'SET_CONTENT_MODE', payload: 'new' })}>
                    New Content
                </button>
                <button className={contentMode === 'update' ? 'active' : ''} onClick={() => dispatch({ type: 'SET_CONTENT_MODE', payload: 'update' })}>
                    Update Existing Content
                </button>
            </div>
            
            {contentMode === 'update' && (
                <ExistingContentTable
                    state={state}
                    dispatch={dispatch}
                    onGenerateContent={onGenerateContent}
                    onGenerateAll={onGenerateAll}
                    onFetchExistingPosts={onFetchExistingPosts}
                />
            )}

            {contentMode === 'new' && (
                 <NewContentHub
                    onGenerate={onGenerateContent}
                    isGenerating={loading}
                    onGenerateIdeas={onGenerateTopicIdeas}
                    isGeneratingTopics={isGeneratingTopics}
                    suggestedTopics={suggestedTopics}
                 />
            )}
        </div>
    );
};

const ReviewModal = ({ state, dispatch, onPublish, onClose }) => {
    const { posts, loading, publishingStatus, currentReviewIndex } = state;
    const [activeTab, setActiveTab] = useState('editor');
    const currentPost = posts[currentReviewIndex];
    
    useEffect(() => {
        setActiveTab('editor');
    }, [currentReviewIndex]);
    
    if (!currentPost) return null;
    
    const updatePostField = (field, value) => {
        dispatch({ type: 'UPDATE_POST_FIELD', payload: { index: currentReviewIndex, field, value } });
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>
                
                 {posts.length > 1 && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                        <button className="btn btn-small" onClick={() => dispatch({type: 'SET_REVIEW_INDEX', payload: Math.max(0, currentReviewIndex - 1)})} disabled={currentReviewIndex === 0}>Previous</button>
                        <span>Viewing {currentReviewIndex + 1} of {posts.length}</span>
                        <button className="btn btn-small" onClick={() => dispatch({type: 'SET_REVIEW_INDEX', payload: Math.min(posts.length - 1, currentReviewIndex + 1)})} disabled={currentReviewIndex === posts.length - 1}>Next</button>
                    </div>
                )}
                
                <div className="review-tabs">
                    <button className={`tab-btn ${activeTab === 'editor' ? 'active' : ''}`} onClick={() => setActiveTab('editor')}>Editor</button>
                    <button className={`tab-btn ${activeTab === 'seo' ? 'active' : ''}`} onClick={() => setActiveTab('seo')}>SEO</button>
                    <button className={`tab-btn ${activeTab === 'preview' ? 'active' : ''}`} onClick={() => setActiveTab('preview')}>Live Preview</button>
                </div>

                <div className="tab-content">
                    {activeTab === 'editor' && (
                        <>
                            <div className="form-group"><label htmlFor="postTitle">Post Title (H1)</label><input type="text" id="postTitle" value={currentPost.title || ''} onChange={e => updatePostField('title', e.target.value)} /></div>
                            <div className="form-group"><label htmlFor="content">HTML Content</label><textarea id="content" value={currentPost.content || ''} onChange={e => updatePostField('content', e.target.value)}></textarea></div>
                        </>
                    )}
                    {activeTab === 'seo' && (
                        <>
                            <div className="form-group"><div className="label-wrapper"><label htmlFor="metaTitle">Meta Title</label><span className="char-counter">{String(currentPost.metaTitle || '').length} / 60</span></div><input type="text" id="metaTitle" value={currentPost.metaTitle || ''} onChange={e => updatePostField('metaTitle', e.target.value)} /></div>
                            <div className="form-group"><div className="label-wrapper"><label htmlFor="metaDescription">Meta Description</label><span className="char-counter">{String(currentPost.metaDescription || '').length} / 160</span></div><textarea id="metaDescription" className="meta-description-input" value={currentPost.metaDescription || ''} onChange={e => updatePostField('metaDescription', e.target.value)} /></div>
                        </>
                    )}
                    {activeTab === 'preview' && (
                        <div className="live-preview">
                            <h1>{currentPost.title}</h1>
                            <div dangerouslySetInnerHTML={{ __html: currentPost.content }} />
                        </div>
                    )}
                </div>

                <div className="button-group">
                    <button className="btn btn-secondary" onClick={onClose}>Back to Selection</button>
                    <button className="btn" onClick={() => onPublish(currentPost)} disabled={loading}>{loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : `Publish to WordPress`}</button>
                </div>
                 {publishingStatus[String(currentPost.id)] && (
                    <div className={`result ${publishingStatus[String(currentPost.id)].success ? 'success' : 'error'}`}>
                        {publishingStatus[String(currentPost.id)].message}
                        {publishingStatus[String(currentPost.id)].link && <>&nbsp;<a href={publishingStatus[String(currentPost.id)].link} target="_blank" rel="noopener noreferrer">View Post</a></>}
                    </div>
                )}
            </div>
        </div>
    );
};

const Footer = () => (
    <footer className="app-footer">
        <p>&copy; {new Date().getFullYear()} <a href="https://affiliatemarketingforsuccess.com" target="_blank" rel="noopener noreferrer">Affiliate Marketing For Success</a>. All Rights Reserved.</p>
        <p>Powered by the AI Content Engine</p>
    </footer>
);

const initialState = {
    currentStep: 1,
    wpUrl: '', wpUser: '', wpPassword: '', sitemapUrl: '',
    posts: [],
    sitemapUrls: [] as string[],
    loading: false, error: null,
    aiProvider: 'gemini',
    apiKeys: { gemini: '', openai: '', anthropic: '', openrouter: '' },
    keyStatus: { gemini: 'unknown', openai: 'unknown', anthropic: 'unknown', openrouter: 'unknown' },
    openRouterModel: 'google/gemini-flash-1.5',
    openRouterModels: ['google/gemini-flash-1.5', 'openai/gpt-4o', 'anthropic/claude-3-haiku'],
    contentMode: 'new',
    publishingStatus: {} as { [key: string]: { success: boolean, message: string, link?: string } },
    generationStatus: {} as { [key: string]: 'idle' | 'generating' | 'done' | 'error' }, // { postId: 'idle' | 'generating' | 'done' | 'error' }
    bulkGenerationProgress: { current: 0, total: 0, visible: false },
    currentReviewIndex: 0,
    isReviewModalOpen: false,
    selectedPostIds: new Set(),
    searchTerm: '',
    sortConfig: { key: 'modified', direction: 'asc' },
    isGeneratingTopics: false,
    suggestedTopics: [] as { title: string; description: string }[],
};

function reducer(state, action) {
    switch (action.type) {
        case 'SET_STEP': return { ...state, currentStep: action.payload };
        case 'SET_FIELD': return { ...state, [action.payload.field]: action.payload.value };
        case 'SET_API_KEY': return { ...state, apiKeys: { ...state.apiKeys, [action.payload.provider]: action.payload.key }, keyStatus: { ...state.keyStatus, [action.payload.provider]: 'validating' } };
        case 'SET_AI_PROVIDER': return { ...state, aiProvider: action.payload };
        case 'SET_KEY_STATUS': return { ...state, keyStatus: { ...state.keyStatus, [action.payload.provider]: action.payload.status } };
        case 'FETCH_START': return { ...state, loading: true, error: null };
        case 'FETCH_SITEMAP_SUCCESS': return { ...state, loading: false, posts: [], sitemapUrls: action.payload.sitemapUrls, currentStep: 2, contentMode: 'new', generationStatus: {}, selectedPostIds: new Set(), suggestedTopics: [] };
        case 'FETCH_EXISTING_POSTS_SUCCESS': return { ...state, loading: false, posts: action.payload, generationStatus: {}, selectedPostIds: new Set(), searchTerm: '', sortConfig: { key: 'modified', direction: 'asc' } };
        case 'FETCH_ERROR': return { ...state, loading: false, error: action.payload };
        case 'SET_GENERATION_STATUS': return { ...state, generationStatus: { ...state.generationStatus, [String(action.payload.postId)]: action.payload.status } };
        case 'GENERATE_SINGLE_POST_SUCCESS': return { ...state, posts: state.posts.map(p => String(p.id) === String(action.payload.id) ? action.payload : p) };
        case 'ADD_GENERATED_POST_AND_REVIEW': {
            const newPosts = [...state.posts, action.payload];
            return {
                ...state,
                posts: newPosts,
                loading: false,
                isReviewModalOpen: true,
                currentReviewIndex: newPosts.length - 1,
            };
        }
        case 'UPDATE_POST_FIELD': return { ...state, posts: state.posts.map((post, index) => index === action.payload.index ? { ...post, [action.payload.field]: action.payload.value } : post) };
        case 'SET_CONTENT_MODE': return { ...state, contentMode: action.payload, posts: [], error: null, generationStatus: {}, selectedPostIds: new Set(), searchTerm: '', suggestedTopics: [] };
        case 'PUBLISH_START': return { ...state, loading: true };
        case 'PUBLISH_SUCCESS': case 'PUBLISH_ERROR': return { ...state, loading: false, publishingStatus: { ...state.publishingStatus, [String(action.payload.postId)]: { success: action.payload.success, message: action.payload.message, link: action.payload.link } } };
        case 'LOAD_CONFIG': return { ...state, ...action.payload };
        case 'SET_REVIEW_INDEX': return { ...state, currentReviewIndex: action.payload };
        case 'OPEN_REVIEW_MODAL': return { ...state, isReviewModalOpen: true, currentReviewIndex: action.payload };
        case 'CLOSE_REVIEW_MODAL': return { ...state, isReviewModalOpen: false };
        case 'TOGGLE_POST_SELECTION': {
            const newSelection = new Set(state.selectedPostIds);
            if (newSelection.has(action.payload)) {
                newSelection.delete(action.payload);
            } else {
                newSelection.add(action.payload);
            }
            return { ...state, selectedPostIds: newSelection };
        }
        case 'SELECT_ALL_VISIBLE': {
            const newSelection = new Set(state.selectedPostIds);
            const allVisibleIds = action.payload;
            const allCurrentlySelected = allVisibleIds.length > 0 && allVisibleIds.every(id => newSelection.has(id));
            if (allCurrentlySelected) {
                allVisibleIds.forEach(id => newSelection.delete(id));
            } else {
                allVisibleIds.forEach(id => newSelection.add(id));
            }
            return { ...state, selectedPostIds: newSelection };
        }
        case 'DESELECT_ALL': return { ...state, selectedPostIds: new Set() };
        case 'SET_SEARCH_TERM': return { ...state, searchTerm: action.payload };
        case 'SET_SORT_CONFIG': return { ...state, sortConfig: action.payload };
        case 'BULK_GENERATE_START': return { ...state, bulkGenerationProgress: { current: 0, total: action.payload, visible: true } };
        case 'BULK_GENERATE_PROGRESS': return { ...state, bulkGenerationProgress: { ...state.bulkGenerationProgress, current: state.bulkGenerationProgress.current + 1 } };
        case 'BULK_GENERATE_COMPLETE': return { ...state, bulkGenerationProgress: { current: 0, total: 0, visible: false } };
        case 'GENERATE_TOPICS_START': return { ...state, isGeneratingTopics: true, error: null, suggestedTopics: [] };
        case 'GENERATE_TOPICS_SUCCESS': return { ...state, isGeneratingTopics: false, suggestedTopics: action.payload };
        case 'GENERATE_TOPICS_ERROR': return { ...state, isGeneratingTopics: false, error: action.payload };
        default: throw new Error(`Unhandled action type: ${action.type}`);
    }
}

const PROMOTIONAL_LINKS = [
    'https://affiliatemarketingforsuccess.com/blog/','https://affiliatemarketingforsuccess.com/seo/write-meta-descriptions-that-convert/','https://affiliatemarketingforsuccess.com/blogging/winning-content-strategy/','https://affiliatemarketingforsuccess.com/review/copy-ai-review/','https://affiliatemarketingforsuccess.com/how-to-start/how-to-choose-a-web-host/','https://affiliatemarketingforsuccess.com/ai/detect-ai-writing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/warriorplus-affiliate-program-unlock-lucrative-opportunities/','https://affiliatemarketingforsuccess.com/affiliate-marketing/understanding-what-is-pay-per-call-affiliate-marketing/','https://affiliatemarketingforsuccess.com/ai/how-chatbot-can-make-you-money/','https://affiliatemarketingforsuccess.com/info/influencer-marketing-sales/','https://affiliatemarketingforsuccess.com/ai/the-power-of-large-language-models/','https://affiliatemarketingforsuccess.com/how-to-start/10-simple-steps-to-build-your-website-a-beginners-guide/','https://affiliatemarketingforsuccess.com/blogging/sustainable-content/','https://affiliatemarketingforsuccess.com/affiliate-marketing/best-discounts-on-black-friday/','https://affiliatemarketingforsuccess.com/seo/website-architecture-that-drives-conversions/','https://affiliatemarketingforsuccess.com/blogging/how-to-create-evergreen-content/','https://affiliatemarketingforsuccess.com/email-marketing/email-marketing-benefits/','https://affiliatemarketingforsuccess.com/blogging/promote-your-blog-to-increase-traffic/','https://affiliatemarketingforsuccess.com/ai/discover-the-power-of-chatgpt/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-with-personalized-recommendations/','https://affiliatemarketingforsuccess.com/seo/benefits-of-an-effective-seo-strategy/','https://affiliatemarketingforsuccess.com/ai/what-is-ai-prompt-engineering/','https://affiliatemarketingforsuccess.com/affiliate-marketing/successful-in-affiliate-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/join-the-best-affiliate-networks/','https://affiliatemarketingforsuccess.com/affiliate-marketing/beginners-guide-to-affiliate-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/high-ticket-affiliate-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/enhance-your-affiliate-marketing-content/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-do-affiliate-marketing-on-shopify/','https://affiliatemarketingforsuccess.com/affiliate-marketing/discover-why-affiliate-marketing-is-the-best-business-model/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-affiliate-marketing-helps-you-to-become-an-influencer/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-affiliate-marketing-on-blog/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-networks/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-create-a-landing-page-for-affiliate-marketing/','https://affiliatemarketingforsuccess.com/review/scalenut-review/','https://affiliatemarketingforsuccess.com/seo/how-to-improve-your-content-marketing-strategy-in-2025/','https://affiliatemarketingforsuccess.com/ai/startup-success-with-chatgpt/','https://affiliatemarketingforsuccess.com/blogging/market-your-blog-the-right-way/','https://affiliatemarketingforsuccess.com/ai/surfer-seo-alternatives/','https://affiliatemarketingforsuccess.com/ai/avoid-ai-detection/','https://affiliatemarketingforsuccess.com/seo/optimize-your-off-page-seo-strategy/','https://affiliatemarketingforsuccess.com/ai/chatgpt-alternative/','https://affiliatemarketingforsuccess.com/seo/build-an-effective-seo-strategy/','https://affiliatemarketingforsuccess.com/email-marketing/understanding-email-marketing/','https://affiliatemarketingforsuccess.com/ai/write-handwritten-assignments/','https://affiliatemarketingforsuccess.com/ai/prompt-engineering-secrets/','https://affiliatemarketingforsuccess.com/seo/boost-your-organic-ranking/','https://affiliatemarketingforsuccess.com/seo/how-to-use-google-my-business-to-improve-your-blogs-local-seo/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-tips-for-beginners/','https://affiliatemarketingforsuccess.com/ai/chatgpt-occupation-prompts/','https://affiliatemarketingforsuccess.com/ai/perplexity-copilot/','https://affiliatemarketingforsuccess.com/ai/agility-writer-vs-autoblogging-ai/','https://affiliatemarketingforsuccess.com/ai/split-testing-perplexity-pages-affiliate-sales/','https://affiliatemarketingforsuccess.com/ai/perplexity-ai-affiliate-funnels-automation/','https://affiliatemarketingforsuccess.com/ai/ai-content-detectors-reliability/','https://affiliatemarketingforsuccess.com/ai/google-bard-bypass-detection/','https://affiliatemarketingforsuccess.com/ai/teachers-detect-gpt-4/','https://affiliatemarketingforsuccess.com/ai/how-to-write-with-perplexity-ai/','https://affiliatemarketingforsuccess.com/ai/turnitin-ai-detection-accuracy/','https://affiliatemarketingforsuccess.com/ai/undetectable-ai-alternatives/','https://affiliatemarketingforsuccess.com/ai/perplexity-jailbreak-prompts-2/','https://affiliatemarketingforsuccess.com/affiliate-marketing/earn-generous-commissions-with-walmart-affiliate-program/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-increase-your-affiliate-marketing-conversion-rate/','https://affiliatemarketingforsuccess.com/ai/how-chat-gpt-will-change-education/','https://affiliatemarketingforsuccess.com/email-marketing/getresponse-review-2025/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-create-an-affiliate-marketing-strategy/','https://affiliatemarketingforsuccess.com/ai/perplexity-model/','https://affiliatemarketingforsuccess.com/email-marketing/proven-ways-to-grow-your-email-list/','https://affiliatemarketingforsuccess.com/ai/undetectable-ai/','https://affiliatemarketingforsuccess.com/review/use-fiverr-gigs-to-boost-your-business/','https://affiliatemarketingforsuccess.com/seo/google-ranking-factors/','https://affiliatemarketingforsuccess.com/ai/how-chat-gpt-is-different-from-google/','https://affiliatemarketingforsuccess.com/blogging/a-guide-to-copyediting-vs-copywriting/','https://affiliatemarketingforsuccess.com/email-marketing/craft-irresistible-email-newsletters/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-on-instagram/','https://affiliatemarketingforsuccess.com/ai/integrate-perplexity-ai-affiliate-tech-stack/','https://affiliatemarketingforsuccess.com/ai/affiliate-marketing-perplexity-ai-future/','https://affiliatemarketingforsuccess.com/blogging/increase-domain-authority-quickly/','https://affiliatemarketingforsuccess.com/review/wp-rocket-boost-wordpress-performance/','https://affiliatemarketingforsuccess.com/affiliate-marketing/shein-affiliate-program-usa-fashionable-earnings-await-you/','https://affiliatemarketingforsuccess.com/ai/auto-ai-transforming-industries-with-automation/','https://affiliatemarketingforsuccess.com/ai/is-turnitin-free/','https://affiliatemarketingforsuccess.com/review/getresponse-vs-clickfunnels/','https://affiliatemarketingforsuccess.com/ai/autoblogging-ai-review/','https://affiliatemarketingforsuccess.com/tools/affiliate-link-generator/','https://affiliatemarketingforsuccess.com/ai/chatgpt-creative-writing-prompts/','https://affiliatemarketingforsuccess.com/ai/undetectable-ai-review/','https://affiliatemarketingforsuccess.com/ai/best-ai-detector/','https://affiliatemarketingforsuccess.com/ai/ai-future-of-seo/','https://affiliatemarketingforsuccess.com/review/clickfunnels-review/','https://affiliatemarketingforsuccess.com/ai/chatgpt-plagiarize/','https://affiliatemarketingforsuccess.com/ai/turnitin-detect-quillbot-paraphrasing/','https://affiliatemarketingforsuccess.com/ai/use-turnitin-checker/','https://affiliatemarketingforsuccess.com/ai/turnitin-read-images/','https://affiliatemarketingforsuccess.com/ai/turnitin-ai-detection-free/','https://affiliatemarketingforsuccess.com/ai/jobs-in-danger-due-to-gpt-4/','https://affiliatemarketingforsuccess.com/ai/surfer-ai-review/','https://affiliatemarketingforsuccess.com/tools/content-idea-generator/','https://affiliatemarketingforsuccess.com/review/getresponse-vs-mailchimp/','https://affiliatemarketingforsuccess.com/ai/turnitin-plagiarism/','https://affiliatemarketingforsuccess.com/email-marketing/getresponse-vs-tinyemail/','https://affiliatemarketingforsuccess.com/affiliate-marketing/struggling-with-wordpress/','https://affiliatemarketingforsuccess.com/ai/learn-prompt-engineering/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-promote-affiliate-products-without-a-website/','https://affiliatemarketingforsuccess.com/ai/chatgpt-playground/','https://affiliatemarketingforsuccess.com/ai/chatgpt-api/','https://affiliatemarketingforsuccess.com/review/frase-review-2025/','https://affiliatemarketingforsuccess.com/review/seowriting-ai-review/','https://affiliatemarketingforsuccess.com/tools/seo-keyword-research-tool/','https://affiliatemarketingforsuccess.com/tools/affiliate-program-comparison-tool/','https://affiliatemarketingforsuccess.com/review/writesonic-review/','https://affiliatemarketingforsuccess.com/blogging/content-marketing-must-educate-and-convert-the-customer/','https://affiliatemarketingforsuccess.com/blogging/how-to-successfully-transition-into-copywriting/','https://affiliatemarketingforsuccess.com/blogging/how-to-use-new-methods-to-capture-leads/','https://affiliatemarketingforsuccess.com/blogging/update-old-blog-content/','https://affiliatemarketingforsuccess.com/review/frase-io-vs-quillbot/','https://affiliatemarketingforsuccess.com/blogging/testimonials-as-blog-content-in-2024/','https://affiliatemarketingforsuccess.com/blogging/overcoming-blog-stagnation/','https://affiliatemarketingforsuccess.com/seo/web-positioning-in-google/','https://affiliatemarketingforsuccess.com/blogging/the-blogging-lifestyle/','https://affiliatemarketingforsuccess.com/review/bramework-review/','https://affiliatemarketingforsuccess.com/seo/how-will-voice-search-impact-your-seo-strategy/','https://affiliatemarketingforsuccess.com/how-to-start/how-to-succeed-in-email-marketing/','https://affiliatemarketingforsuccess.com/review/spreadsimple-review/','https://affiliatemarketingforsuccess.com/ai/boost-affiliate-earnings-perplexity-ai/','https://affiliatemarketingforsuccess.com/tools/script-timer-tool/','https://affiliatemarketingforsuccess.com/ai/agility-writer-review/','https://affiliatemarketingforsuccess.com/review/inkforall-review-2024/','https://affiliatemarketingforsuccess.com/web-hosting/web-hosting-comparison/','https://affiliatemarketingforsuccess.com/ai/is-chatgpt-down-what-happened-and-how-to-fix-it/','https://affiliatemarketingforsuccess.com/review/namehero-hosting-review/','https://affiliatemarketingforsuccess.com/review/katteb-review/','https://affiliatemarketingforsuccess.com/blogging/wordpress-blogging-tips/','https://affiliatemarketingforsuccess.com/review/neuronwriter-review/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-quickly-can-i-make-money-with-affiliate-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/step-by-step-in-affiliate-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/the-costs-to-start-affiliate-marketing/','https://affiliatemarketingforsuccess.com/blogging/grow-your-affiliate-marketing-blog/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-niche-selection-mistakes/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-reviews/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-tools/','https://affiliatemarketingforsuccess.com/affiliate-marketing/digital-marketing-definition/','https://affiliatemarketingforsuccess.com/affiliate-marketing/build-an-affiliate-marketing-business/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-success/','https://affiliatemarketingforsuccess.com/affiliate-marketing/best-ai-affiliate-niches/','https://affiliatemarketingforsuccess.com/affiliate-marketing/the-concepts-of-digital-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/building-an-affiliate-marketing-website/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-affiliate-marketing-works/','https://affiliatemarketingforsuccess.com/affiliate-marketing/maximize-your-startup-potential-leverage-chatgpt-for-startups-with-expert-chatgpt-prompts/','https://affiliatemarketingforsuccess.com/review/grammarly-premium-review-leveradge-your-writing/','https://affiliatemarketingforsuccess.com/blogging/how-to-position-your-blog/','https://affiliatemarketingforsuccess.com/blogging/how-to-quickly-generate-leads/','https://affiliatemarketingforsuccess.com/blogging/what-is-the-best-structure-of-a-blog-post/','https://affiliatemarketingforsuccess.com/ai/chatgpt-has-changed-seo-forever/','https://affiliatemarketingforsuccess.com/blogging/8-tips-for-successful-copywriting/','https://affiliatemarketingforsuccess.com/blogging/why-do-blogs-fail/','https://affiliatemarketingforsuccess.com/ai/copywriting-frameworks/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-long-it-takes-to-become-an-affiliate-marketer/','https://affiliatemarketingforsuccess.com/make-money-online/business-models-to-make-money-online/','https://affiliatemarketingforsuccess.com/review/blogify-ai-review/','https://affiliatemarketingforsuccess.com/review/wpx-hosting-review/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-for-online-business/','https://affiliatemarketingforsuccess.com/review/kinsta-wordpress-hosting-review/','https://affiliatemarketingforsuccess.com/review/marketmuse-review/','https://affiliatemarketingforsuccess.com/blogging/how-to-analyze-your-blogs-user-behavior-metrics/','https://affiliatemarketingforsuccess.com/ai/prompt-engineering-examples/','https://affiliatemarketingforsuccess.com/blogging/how-to-increase-your-online-sales-at-christmas/','https://affiliatemarketingforsuccess.com/blogging/keys-to-creating-successful-content-on-your-blog/','https://affiliatemarketingforsuccess.com/review/writesonic-vs-seowriting-ai/','https://affiliatemarketingforsuccess.com/ai/perplexity-ai-revolutionize-affiliate-strategy/','https://affiliatemarketingforsuccess.com/ai/perplexity-ai-affiliate-marketing/','https://affiliatemarketingforsuccess.com/blogging/create-seo-friendly-blog-posts/','https://affiliatemarketingforsuccess.com/ai/chatgpt-for-education/','https://affiliatemarketingforsuccess.com/make-money-online/what-is-the-profile-of-a-successful-online-entrepreneur/','https://affiliatemarketingforsuccess.com/ai/bard-vs-chatgpt-vs-grok/','https://affiliatemarketingforsuccess.com/blogging/automate-your-blog-with-artificial-intelligence/','https://affiliatemarketingforsuccess.com/info/how-to-screenshot-on-chromebook/','https://affiliatemarketingforsuccess.com/ai/chatgpt-detected-by-safeassign/','https://affiliatemarketingforsuccess.com/ai/turnitin-vs-grammarly/','https://affiliatemarketingforsuccess.com/affiliate-marketing/what-are-impressions-in-advertising/','https://affiliatemarketingforsuccess.com/blogging/11-things-to-outsource-as-a-blogger-for-more-time-and-efficiency/','https://affiliatemarketingforsuccess.com/email-marketing/email-list-for-affiliate-marketing/','https://affiliatemarketingforsuccess.com/review/copy-ai-vs-katteb/','https://affiliatemarketingforsuccess.com/how-to-start/google-ranking-factors-seo-strategy/','https://affiliatemarketingforsuccess.com/make-money-online/how-to-make-money-writing-articles-online/','https://affiliatemarketingforsuccess.com/blogging/best-topics-on-your-digital-marketing-blog/','https://affiliatemarketingforsuccess.com/web-hosting/digitalocean-review/','https://affiliatemarketingforsuccess.com/blogging/top-challenges-a-blogger-faces/','https://affiliatemarketingforsuccess.com/blogging/how-to-boost-the-ranking-of-an-existing-page-on-search-engines/','https://affiliatemarketingforsuccess.com/blogging/create-your-personal-blog/','https://affiliatemarketingforsuccess.com/ai/chatgpt-vs-competing-language-models/','https://affiliatemarketingforsuccess.com/info/paraphrase-text-using-nlp/','https://affiliatemarketingforsuccess.com/review/pictory-ai-review/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-track-and-measure-your-affiliate-marketing-performance/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-use-seo-for-affiliate-marketing/','https://affiliatemarketingforsuccess.com/seo/mastering-seo-best-practices/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-much-time-it-takes-to-earn-from-affiliate-marketing/','https://affiliatemarketingforsuccess.com/blogging/google-pagespeed-insights/','https://affiliatemarketingforsuccess.com/blogging/the-imposter-syndrome/','https://affiliatemarketingforsuccess.com/blogging/lead-nurturing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-vs-dropshipping/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-make-money-with-affiliate-marketing/','https://affiliatemarketingforsuccess.com/seo/the-importance-of-seo-for-your-blog/','https://affiliatemarketingforsuccess.com/how-to-start/criteria-for-profitable-affiliate-niches/','https://affiliatemarketingforsuccess.com/ai/chatgpt-give-same-answers/','https://affiliatemarketingforsuccess.com/affiliate-marketing/why-affiliate-marketers-fail/','https://affiliatemarketingforsuccess.com/ai/winston-detect-quillbot/','https://affiliatemarketingforsuccess.com/ai/quillbot-bypass-ai-detection/','https://affiliatemarketingforsuccess.com/ai/how-chatgpt-gets-information/','https://affiliatemarketingforsuccess.com/email-marketing/effective-email-marketing-strategies/','https://affiliatemarketingforsuccess.com/ai/semantic-clustering-in-seo/','https://affiliatemarketingforsuccess.com/ai/semantic-clustering-tools/','https://affiliatemarketingforsuccess.com/blogging/how-to-write-niche-specific-content/','https://affiliatemarketingforsuccess.com/make-money-online/optimize-your-sales-funnel/','https://affiliatemarketingforsuccess.com/affiliate-marketing/best-affiliate-marketing-niches-2025/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-start-an-affiliate-marketing-blog/','https://affiliatemarketingforsuccess.com/blogging/how-to-setup-the-basic-seo-technical-foundations-for-your-blog/','https://affiliatemarketingforsuccess.com/blogging/long-term-content-strategy/','https://affiliatemarketingforsuccess.com/ai/how-chatgpt-works/','https://affiliatemarketingforsuccess.com/ai/prompt-engineering-nlp/','https://affiliatemarketingforsuccess.com/ai/prompt-engineering-course/','https://affiliatemarketingforsuccess.com/ai/prompt-engineering-ai-art/','https://affiliatemarketingforsuccess.com/review/semrush-review-2025/','https://affiliatemarketingforsuccess.com/affiliate-marketing/top-10-affiliate-marketing-trends-in-2025/','https://affiliatemarketingforsuccess.com/affiliate-marketing/launch-affiliate-business-ai-tools/','https://affiliatemarketingforsuccess.com/blogging/monetize-your-blog-proven-strategies/','https://affiliatemarketingforsuccess.com/ai/ethical-implications-of-ai/','https://affiliatemarketingforsuccess.com/web-hosting/siteground-web-hosting-review-2025/','https://affiliatemarketingforsuccess.com/ai/deepseek-r1-vs-chatgpt/','https://affiliatemarketingforsuccess.com/ai/prompt-engineering-jobs/','https://affiliatemarketingforsuccess.com/ai/perplexity-ai/','https://affiliatemarketingforsuccess.com/review/the-ultimate-jasper-ai-review/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-use-social-media-for-affiliate-marketing/','https://affiliatemarketingforsuccess.com/ai/chatgpt-use-cases/','https://affiliatemarketingforsuccess.com/seo/the-importance-of-keywords-research/','https://affiliatemarketingforsuccess.com/ai/ai-prompt-writing/','https://affiliatemarketingforsuccess.com/blogging/what-is-copywriting-promotes-advertises-or-entertains/','https://affiliatemarketingforsuccess.com/blogging/how-to-write-a-high-ranking-blog-post/','https://affiliatemarketingforsuccess.com/affiliate-marketing/generative-ai/','https://affiliatemarketingforsuccess.com/how-to-start/how-to-register-a-domain-name/','https://affiliatemarketingforsuccess.com/chatgpt-prompts/chatgpt-prompts-for-marketing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-many-affiliate-programs-should-i-join-guide/','https://affiliatemarketingforsuccess.com/how-to-start/top-10-pro-tips-for-choosing-affiliate-marketing-programs/','https://affiliatemarketingforsuccess.com/affiliate-marketing/optimize-your-affiliate-marketing-website-for-seo/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-use-youtube-for-affiliate-marketing/','https://affiliatemarketingforsuccess.com/ai/ai-affiliate-marketing-strategies-2025/','https://affiliatemarketingforsuccess.com/review/quillbot-review/','https://affiliatemarketingforsuccess.com/how-to-start/how-to-choose-your-niche/','https://affiliatemarketingforsuccess.com/affiliate-marketing/how-to-make-money-with-amazon-affiliate-marketing/','https://affiliatemarketingforsuccess.com/ai/best-chatgpt-alternatives-for-2025/','https://affiliatemarketingforsuccess.com/how-to-start/most-suitable-affiliate-program/','https://affiliatemarketingforsuccess.com/seo/seo-writing-a-complete-guide-to-seo-writing/','https://affiliatemarketingforsuccess.com/how-to-start/the-truth-about-web-hosting/','https://affiliatemarketingforsuccess.com/ai/chatgpt-prompt-engineering/','https://affiliatemarketingforsuccess.com/blogging/storytelling-in-content-marketing/','https://affiliatemarketingforsuccess.com/tools/email-marketing-template-generator/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-mistakes/','https://affiliatemarketingforsuccess.com/seo/keyword-stemming/','https://affiliatemarketingforsuccess.com/ai/multimodal-ai-models-guide/','https://affiliatemarketingforsuccess.com/ai/large-language-models-comparison-2025/','https://affiliatemarketingforsuccess.com/ai/gpt-4o-vs-gemini/','https://affiliatemarketingforsuccess.com/ai/multimodal-prompt-engineering/','https://affiliatemarketingforsuccess.com/ai/claude-4-guide/','https://affiliatemarketingforsuccess.com/seo/programmatic-seo/','https://affiliatemarketingforsuccess.com/blogging/blogging-mistakes-marketers-make/','https://affiliatemarketingforsuccess.com/seo/why-your-current-seo-strategy-is-failing/','https://affiliatemarketingforsuccess.com/blogging/how-to-brand-storytelling/','https://affiliatemarketingforsuccess.com/seo/doing-an-seo-audit/','https://affiliatemarketingforsuccess.com/tools/commission-calculator/','https://affiliatemarketingforsuccess.com/blogging/essential-tools-for-a-blogger/','https://affiliatemarketingforsuccess.com/blogging/types-of-evergreen-content/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-strategies/','https://affiliatemarketingforsuccess.com/review/cloudways-review-2025/','https://affiliatemarketingforsuccess.com/affiliate-marketing/the-power-of-ai-in-seo/','https://affiliatemarketingforsuccess.com/ai/artificial-intelligence-machine-learning-revolutionizing-the-future/','https://affiliatemarketingforsuccess.com/affiliate-marketing/keys-to-successful-affiliate-marketing/','https://affiliatemarketingforsuccess.com/seo/improve-your-ranking-in-seo/','https://affiliatemarketingforsuccess.com/blogging/reduce-bounce-rate/','https://affiliatemarketingforsuccess.com/blogging/what-is-a-creative-copywriter/','https://affiliatemarketingforsuccess.com/ai/chatgpt4-vs-gemini-pro-in-blog-writing/','https://affiliatemarketingforsuccess.com/blogging/build-a-blogging-business-from-scratch/','https://affiliatemarketingforsuccess.com/affiliate-marketing/the-ultimate-guide-to-affiliate-marketing/','https://affiliatemarketingforsuccess.com/email-marketing/convertkit-pricing/','https://affiliatemarketingforsuccess.com/affiliate-marketing/best-affiliate-products-to-promote/','https://affiliatemarketingforsuccess.com/make-money-online/how-to-make-money-with-clickbank-the-ultimate-guide/','https://affiliatemarketingforsuccess.com/seo/link-building-strategies/','https://affiliatemarketingforsuccess.com/affiliate-marketing/affiliate-marketing-on-pinterest/','https://affiliatemarketingforsuccess.com/blogging/blog-monetization-strategies/','https://affiliatemarketingforsuccess.com/affiliate-marketing/why-is-affiliate-marketing-so-hard/','https://affiliatemarketingforsuccess.com/ai/originality-ai-review/','https://affiliatemarketingforsuccess.com/ai/how-chatbot-helps-developers/','https://affiliatemarketingforsuccess.com/info/how-to-make-a-social-media-marketing-plan/','https://affiliatemarketingforsuccess.com/blogging/countless-benefits-of-blogging/','https://affiliatemarketingforsuccess.com/ai/the-anthropic-prompt-engineer/','https://affiliatemarketingforsuccess.com/ai/nvidia-ai/','https://affiliatemarketingforsuccess.com/chatgpt-prompts/awesome-chatgpt-prompts/','https://affiliatemarketingforsuccess.com/ai/ai-powered-semantic-clustering/','https://affiliatemarketingforsuccess.com/ai/semantic-clustering-techniques/','https://affiliatemarketingforsuccess.com/ai/benefits-of-semantic-clustering/',
];


const App = () => {
    const [state, dispatch] = useReducer(reducer, initialState);

    useEffect(() => {
        const savedConfig = localStorage.getItem('wpContentOptimizerConfig');
        if (savedConfig) dispatch({ type: 'LOAD_CONFIG', payload: JSON.parse(savedConfig) });
    }, []);

    const handleValidateKey = useCallback(async (provider, key) => {
        dispatch({ type: 'SET_KEY_STATUS', payload: { provider, status: 'validating' } });
        try {
            if (!key || key.length < 10) throw new Error('Invalid key format');
            await new Promise(res => setTimeout(res, 500)); // Basic validation
            dispatch({ type: 'SET_KEY_STATUS', payload: { provider, status: 'valid' } });
        } catch (error) {
            dispatch({ type: 'SET_KEY_STATUS', payload: { provider, status: 'invalid' } });
        }
    }, []);
    
    const handleFetchSitemap = async (sitemapUrl, saveConfig) => {
        dispatch({ type: 'FETCH_START' });
        if (saveConfig) {
            localStorage.setItem('wpContentOptimizerConfig', JSON.stringify({ wpUrl: state.wpUrl, wpUser: state.wpUser, wpPassword: state.wpPassword, aiProvider: state.aiProvider, apiKeys: state.apiKeys }));
        }
        
        try {
            const allUrls = await parseSitemap(sitemapUrl);
            if (allUrls.length === 0) throw new Error("No URLs found in the sitemap, or the sitemap could not be parsed.");
            
            const uniqueUrls = [...new Set(allUrls)];
            dispatch({ type: 'FETCH_SITEMAP_SUCCESS', payload: { sitemapUrls: uniqueUrls } });

        } catch (error) {
            const message = (error instanceof Error) ? error.message : String(error);
            dispatch({ type: 'FETCH_ERROR', payload: `Error processing sitemap: ${message}. Please verify the URL is correct and accessible.` });
        }
    };

    const handleFetchExistingPosts = async () => {
        dispatch({ type: 'FETCH_START' });
        const { wpUrl, wpUser, wpPassword } = state;
        try {
            const endpoint = `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts?_fields=id,title,link,modified&per_page=50&orderby=modified&order=asc`;
            const headers = new Headers({ 'Authorization': 'Basic ' + btoa(`${wpUser}:${wpPassword}`) });
            const response = await directFetch(endpoint, { headers });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
            }
            const existingPosts = await response.json();
            const formattedPosts = existingPosts.map(p => ({ id: p.id, title: p.title.rendered, url: p.link, modified: p.modified, content: '' }));
            dispatch({ type: 'FETCH_EXISTING_POSTS_SUCCESS', payload: formattedPosts });
        } catch (error) {
            const message = (error instanceof Error) ? error.message : String(error);
            dispatch({ type: 'FETCH_ERROR', payload: `Error fetching existing posts: ${message}` });
        }
    };
    
    const getAiClient = () => {
        const apiKey = state.apiKeys[state.aiProvider];
        switch (state.aiProvider) {
            case 'gemini': return new GoogleGenAI({ apiKey });
            case 'openai': return new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
            case 'anthropic': return new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
            case 'openrouter': return new OpenAI({ baseURL: "https://openrouter.ai/api/v1", apiKey, defaultHeaders: { "HTTP-Referer": "http://localhost:3000", "X-Title": "AI Content Engine" }, dangerouslyAllowBrowser: true });
            default: throw new Error('Unsupported AI provider');
        }
    };

    const handleGenerateTopicIdeas = async () => {
        dispatch({ type: 'GENERATE_TOPICS_START' });
        const topicPrompt = `
You are a master SEO strategist and content planner for a website in the affiliate marketing niche.
Based on the following list of existing article URLs from the website, analyze the site's current topical authority. Your goal is to identify strategic content gaps and propose 5 new, high-impact pillar post ideas that will significantly boost organic traffic and strengthen the website's expertise.

**Analysis of Existing URLs:**
${getRandomSubset(state.sitemapUrls, 50).join('\n')}

**Your Task:**
Generate 5 blog post ideas that meet the following criteria:
1.  **Topical Relevance:** They must be highly relevant to affiliate marketing, SEO, and making money online, complementing the existing content.
2.  **High Traffic Potential:** Target keywords with substantial search volume and a clear path to ranking.
3.  **Pillar Post Quality:** Each topic should be broad and deep enough to be developed into a 1800+ word comprehensive guide.
4.  **Strategic Value:** The ideas should fill content gaps, attract a valuable audience segment, or target a lucrative sub-niche.

**Output Format:**
You MUST return a single, valid JSON object. The object should have a single key "ideas", which is an array of 5 objects. Each object in the array must have these two keys:
- "title": A compelling, SEO-optimized H1 title for the blog post.
- "description": A short, 1-2 sentence description explaining the strategic value of this topic and who it's for.
`;
        try {
            const ai = getAiClient();
            if (state.aiProvider !== 'gemini') {
                throw new Error("Topic idea generation is currently only supported for the Google Gemini provider for best results.");
            }
            const { parsedContent } = await makeResilientAiCall(async () => {
                 const response = await (ai as GoogleGenAI).models.generateContent({
                    model: 'gemini-2.5-flash',
                    contents: topicPrompt,
                    config: {
                        responseMimeType: "application/json",
                        responseSchema: {
                            type: Type.OBJECT,
                            properties: {
                                ideas: {
                                    type: Type.ARRAY,
                                    items: {
                                        type: Type.OBJECT,
                                        properties: {
                                            title: { type: Type.STRING },
                                            description: { type: Type.STRING }
                                        },
                                        required: ["title", "description"]
                                    }
                                }
                            },
                             required: ["ideas"]
                        }
                    }
                });
                const jsonText = extractJson(response.text);
                const data = JSON.parse(jsonText);
                return { parsedContent: data };
            });

            if (!parsedContent.ideas || parsedContent.ideas.length === 0) {
                throw new Error("AI did not return any topic ideas.");
            }
            dispatch({ type: 'GENERATE_TOPICS_SUCCESS', payload: parsedContent.ideas });

        } catch (error) {
            console.error("Topic Generation Error", error);
            const message = (error instanceof Error) ? error.message : String(error);
            dispatch({ type: 'GENERATE_TOPICS_ERROR', payload: `Error generating topic ideas: ${message}` });
        }
    };


    const handleGenerateContent = async (postOrTopic) => {
        const isNewContent = typeof postOrTopic === 'string';
        const postToProcess = isNewContent ? { id: -Date.now(), title: postOrTopic } : postOrTopic;

        if (isNewContent) {
            dispatch({ type: 'FETCH_START' }); // Use loading state for the whole screen
        } else {
            dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'generating' } });
        }

        let internalLinkCandidates: string[] = isNewContent
            ? state.sitemapUrls
            : state.posts.map(p => p.url).filter(Boolean);
        
        const relevantInternalLinks = getRandomSubset(internalLinkCandidates, 50);
        const internalLinksList = relevantInternalLinks.length > 0
            ? relevantInternalLinks.map(url => `- [${slugToTitle(url)}](${url})`).join('\n')
            : '- No internal links available.';
            
        const internalLinksInstruction = `**Internal Linking:** Your primary goal is to include 6-10 highly relevant internal links within the article body. You MUST choose them from the following list of articles from the user's website. Use rich, descriptive anchor text. Do NOT use placeholder links.\n\n**Available Internal Links:**\n${internalLinksList}`;
        
        const referencesInstruction = state.aiProvider === 'gemini' 
            ? `**CRITICAL: Use Google Search:** You MUST use Google Search for up-to-date, authoritative info. A "References" section will be auto-generated from real search results to ensure all links are valid, functional, 200 OK pages.`
            : `**CRITICAL: Add 100% REAL, VERIFIABLE References:** After the conclusion, you MUST add an H2 section titled "References". In this section, provide a bulleted list (\`<ul>\`) of 6-12 links to REAL, CURRENT, and ACCESSIBLE authoritative external sources. Every single link MUST be a fully functional, live URL that resolves to a 200 OK page. Do not invent, guess, or hallucinate URLs. Your absolute top priority is the accuracy and validity of these links.`;

        const topicOrUrl = isNewContent ? postToProcess.title : (postToProcess.url || postToProcess.title);
        let task;

        if (isNewContent) {
            task = "Write the ultimate, SEO-optimized blog post on the topic.";
        } else {
            if (state.aiProvider === 'gemini') {
                task = "Completely rewrite and supercharge the blog post from the URL into a definitive resource. Use your search capabilities to find the latest information and the content at the URL.";
            } else {
                task = `An existing blog post is at the URL below. Your task is to write a completely new, definitive, and supercharged article on the same topic, making it 10x better than the competition. Do NOT try to access the URL; instead, use your general knowledge to create a superior piece of content on the subject matter implied by the URL.`;
            }
        }
        
        const basePrompt = `You are a world-class SEO and content strategist, operating as a definitive expert in the given topic. Your writing style is direct, high-conviction, and packed with actionable value, reminiscent of Alex Hormozi. You challenge conventional wisdom and provide non-obvious insights. Your mission is to produce a comprehensive, 1800+ word pillar blog post that is strategically designed to rank #1 on Google.

**Core Task:** ${task}

**Pillar Post Generation Protocol:**

1.  **Simulated SERP & Gap Analysis:**
    *   Deconstruct the topic: Identify the primary keyword, related LSI keywords, and common "People Also Ask" questions.
    *   Competitor Gap Analysis: Mentally analyze what the top 5 search results for this topic likely cover. Your primary objective is to create "10x content" that addresses the gaps they've missed, providing significantly more value, depth, and unique insights.

2.  **Content & Tone:**
    *   **Expert Persona:** Write with extreme authority. Inject critical thinking, simulated personal experience, and strong, defensible opinions to make the content trustworthy and unique. Avoid generic, fluffy language.
    *   **Readability is Key:** Use short, punchy paragraphs (2-3 sentences max). Utilize bolding for key terms, bullet points (\`<ul>\`), and numbered lists (\`<ol>\`) to make the extensive content scannable and easy to digest.

3.  **Required Article Structure (in this exact order):**
    *   **"Wow" Introduction:** Start with a compelling hook, such as a surprising statistic, a bold contrarian claim, or a relatable pain point to grab the reader's attention immediately.
    *   **Key Takeaways Box:** Immediately after the intro, add an H3 titled "Key Takeaways" inside a \`<div class="key-takeaways">\`. Provide a bulleted list of 6-8 crucial, actionable points from the article.
    *   **Comprehensive Body:** This is the main section. Create a deep-dive exploration of the topic using clear H2 and H3 subheadings. The final article MUST be a pillar post of at least 1800 words. Cover all aspects of the main keyword, LSI terms, and the insights from your gap analysis.
    *   **Internal Linking:** ${internalLinksInstruction}
    *   **FAQ Section:** Include an H2 titled "Frequently Asked Questions" and answer 3-5 relevant questions (inspired by "People Also Ask").
    *   **Conclusion:** Provide a strong, actionable conclusion that summarizes the key points and gives the reader a clear, compelling next step.
    *   **References:** ${referencesInstruction}

4.  **Final JSON Output:**
    *   You MUST return a single, valid JSON object.
    *   The JSON object must have these exact keys: "title" (a compelling, SEO-friendly H1 title), "metaTitle" (50-60 characters), "metaDescription" (150-160 characters), and "content" (the full HTML body of the article).
    *   The "content" string MUST NOT include the main <h1> title, as that will be handled separately.

**${isNewContent ? 'Topic' : 'URL'}:** ${topicOrUrl}`;

        try {
            const ai = getAiClient();
            
            const { parsedContent, geminiResponse } = await makeResilientAiCall(async () => {
                let generatedText: string;
                let response: GenerateContentResponse | null = null;
    
                if (state.aiProvider === 'gemini') {
                    response = await (ai as GoogleGenAI).models.generateContent({ model: 'gemini-2.5-flash', contents: basePrompt, config: { tools: [{ googleSearch: {} }] } });
                    generatedText = response.text;
                } else if (state.aiProvider === 'anthropic') {
                    const anthropicResponse = await (ai as Anthropic).messages.create({ model: 'claude-3-haiku-20240307', max_tokens: 4096, messages: [{ role: 'user', content: basePrompt }] });
                    generatedText = anthropicResponse.content[0].type === 'text' ? anthropicResponse.content[0].text : '';
                } else {
                    const openAiResponse = await (ai as OpenAI).chat.completions.create({ model: state.aiProvider === 'openai' ? 'gpt-4o' : state.openRouterModel, messages: [{ role: 'user', content: basePrompt }], response_format: { type: "json_object" } });
                    generatedText = openAiResponse.choices[0].message.content;
                }
                
                if (!generatedText) {
                    throw new Error("AI returned an empty response.");
                }
                
                const jsonText = extractJson(generatedText);
                const data = JSON.parse(jsonText);
    
                if (!data || !data.content) {
                    throw new Error("AI response is missing required 'content' field.");
                }
    
                return { parsedContent: data, geminiResponse: response };
            });

            let finalContent = parsedContent.content || '';

            const groundingChunks = geminiResponse?.candidates?.[0]?.groundingMetadata?.groundingChunks;
            if (state.aiProvider === 'gemini' && groundingChunks && groundingChunks.length > 0) {
                const uniqueReferences = (groundingChunks || []).reduce((map, chunk) => {
                    if (chunk.web?.uri) {
                        map.set(chunk.web.uri, chunk.web.title || chunk.web.uri);
                    }
                    return map;
                }, new Map<string, string>());

                if (uniqueReferences.size > 0) {
                    let referencesHtml = '<div class="references-section"><h2>References</h2><ul>';
                    uniqueReferences.forEach((title, uri) => {
                        referencesHtml += `<li><a href="${uri}" target="_blank" rel="noopener noreferrer">${title}</a></li>`;
                    });
                    finalContent += referencesHtml + '</ul></div>';
                }
            }
            
            const finalPost = { ...postToProcess, title: parsedContent.title || postToProcess.title, metaTitle: parsedContent.metaTitle || '', metaDescription: parsedContent.metaDescription || '', content: finalContent || '<p>Error: Content generation failed.</p>' };
            
            if (isNewContent) {
                dispatch({ type: 'ADD_GENERATED_POST_AND_REVIEW', payload: finalPost });
            } else {
                dispatch({ type: 'GENERATE_SINGLE_POST_SUCCESS', payload: finalPost });
                dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'done' } });
            }

        } catch (error) {
            console.error("AI Generation Error for", topicOrUrl, error);
            let detailedMessage = "An unknown error occurred.";
            if (error instanceof Error) {
                detailedMessage = error.message;
            } else if (typeof error === 'string') {
                detailedMessage = error;
            }
            
            const errorMessage = (detailedMessage.includes('429')) 
                ? `Rate limit exceeded: ${detailedMessage}` 
                : `Error generating content: ${detailedMessage}`;

            if(isNewContent) {
                dispatch({ type: 'FETCH_ERROR', payload: errorMessage });
            } else {
                 dispatch({ type: 'GENERATE_SINGLE_POST_SUCCESS', payload: { ...postToProcess, content: `<p>Error: ${errorMessage}</p>` } });
                 dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'error' } });
            }
        }
    };

    const handleGenerateAll = async () => {
        const postsToProcess = (state.selectedPostIds.size > 0
            ? state.posts.filter(p => state.selectedPostIds.has(p.id))
            : state.posts)
            .filter(p => {
                const status = state.generationStatus[String(p.id)];
                return status !== 'done' && status !== 'generating';
            });

        if (postsToProcess.length === 0) return;
        
        dispatch({ type: 'BULK_GENERATE_START', payload: postsToProcess.length });

        const progressCallback = () => {
            dispatch({ type: 'BULK_GENERATE_PROGRESS' });
        };
        
        await processPromiseQueue(postsToProcess, handleGenerateContent, progressCallback, 2000);
        
        dispatch({ type: 'BULK_GENERATE_COMPLETE' });
    };


    const handlePublish = async (post) => {
        dispatch({ type: 'PUBLISH_START' });
        const { wpUrl, wpUser, wpPassword } = state;
        try {
            const isUpdate = typeof post.id === 'number' && post.id > 0;
            const endpoint = isUpdate ? `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts/${post.id}` : `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts`;
            const headers = new Headers({ 'Authorization': 'Basic ' + btoa(`${wpUser}:${wpPassword}`), 'Content-Type': 'application/json' });
            const body = JSON.stringify({ title: post.title, content: post.content, status: 'publish', meta: { _yoast_wpseo_title: post.metaTitle, _yoast_wpseo_metadesc: post.metaDescription } });
            const response = await directFetch(endpoint, { method: 'POST', headers, body });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
            }
            const responseData = await response.json();
            dispatch({ type: 'PUBLISH_SUCCESS', payload: { postId: post.id, success: true, message: `Successfully ${isUpdate ? 'updated' : 'published'} "${responseData.title.rendered}"!`, link: responseData.link } });
        } catch (error) {
            const message = (error instanceof Error) ? error.message : String(error);
            dispatch({ type: 'PUBLISH_ERROR', payload: { postId: post.id, success: false, message: `Error publishing post: ${message}` } });
        }
    };
    
    const renderContent = () => {
        switch (state.currentStep) {
            case 1: return <ConfigStep state={state} dispatch={dispatch} onFetchSitemap={handleFetchSitemap} onValidateKey={handleValidateKey} />;
            case 2: return <ContentStep state={state} dispatch={dispatch} onGenerateContent={handleGenerateContent} onFetchExistingPosts={handleFetchExistingPosts} onGenerateAll={handleGenerateAll} onGenerateTopicIdeas={handleGenerateTopicIdeas} />;
            default: return <div>Error: Invalid step.</div>;
        }
    };

    return (
        <>
            <div className="container">
                <div className="app-header">
                     <h1>AI Content Engine</h1>
                </div>
                {state.error && <div className="result error">{state.error}</div>}
                
                <ProgressBar currentStep={state.currentStep} />
                
                {renderContent()}

                 {state.isReviewModalOpen && <ReviewModal state={state} dispatch={dispatch} onPublish={handlePublish} onClose={() => dispatch({ type: 'CLOSE_REVIEW_MODAL' })} />}
            </div>
            <Footer />
        </>
    );
};

ReactDOM.createRoot(document.getElementById('root')!).render(<App />);
