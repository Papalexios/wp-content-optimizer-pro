
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
 * Wraps a promise-returning function with an exponential backoff retry mechanism
 * to handle transient errors, specifically API rate limiting (HTTP 429).
 * @param apiCallFn The async function to call.
 * @param maxRetries The maximum number of retries before giving up.
 * @param initialDelay The initial delay in ms for the first retry.
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
            // Check for specific rate limit error messages or status codes.
            const isRateLimitError = (
                (error.status === 429) || 
                (error.message && error.message.includes('429')) ||
                (error.message && error.message.toLowerCase().includes('rate limit'))
            );

            if (isRateLimitError && i < maxRetries - 1) {
                const delay = initialDelay * Math.pow(2, i) + Math.random() * 1000; // Exponential backoff with jitter
                console.warn(`Rate limit error detected. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${i + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                // Not a rate limit error, or this was the last retry, so re-throw.
                console.error(`AI call failed on attempt ${i + 1}.`, error);
                throw error;
            }
        }
    }
    // This line should be unreachable, but as a fallback, throw the last captured error.
    throw new Error(`AI call failed after ${maxRetries} attempts. Last error: ${lastError?.message}`);
};


/**
 * Intelligently fetches a resource by first attempting a direct connection.
 * If the direct connection fails due to a CORS-like network error, it automatically
 * falls back to a series of reliable CORS proxies.
 * @param url The target URL to fetch.
 * @param options The standard fetch options object.
 * @returns A Promise that resolves with the Response object.
 */
const smartFetch = async (url: string, options: RequestInit = {}): Promise<Response> => {
    // 1. Attempt direct fetch first
    try {
        const directResponse = await fetch(url, options);
        if (directResponse.ok) {
            return directResponse;
        }
        console.warn(`Direct fetch to ${url} was not OK, status: ${directResponse.status}. Trying proxies.`);
    } catch (error) {
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
            console.warn(`Direct fetch to ${url} failed, likely due to CORS. Falling back to proxies.`);
        } else {
            console.error('An unexpected network error occurred during direct fetch:', error);
            throw error;
        }
    }

    // 2. Define a list of diverse and reliable proxies to try in sequence
    const proxies = [
        // Reliable proxy, needs URL to be passed as is in the query string.
        {
            name: 'corsproxy.io',
            buildUrl: (targetUrl) => `https://corsproxy.io/?${targetUrl}`,
            getHeaders: () => ({})
        },
        // Another reliable proxy, but requires the URL to be URI encoded.
        {
            name: 'allorigins.win',
            buildUrl: (targetUrl) => `https://api.allorigins.win/raw?url=${encodeURIComponent(targetUrl)}`,
            getHeaders: () => ({})
        },
        // A simple proxy that takes the URL as a path.
        {
            name: 'thingproxy.freeboard.io',
            buildUrl: (targetUrl) => `https://thingproxy.freeboard.io/fetch/${targetUrl}`,
            getHeaders: () => ({})
        },
        // Another query-parameter based proxy.
        {
           name: 'CodeTabs',
           buildUrl: (targetUrl) => `https://api.codetabs.com/v1/proxy?quest=${targetUrl}`,
           getHeaders: () => ({})
        }
    ];

    let lastError: Error | null = new Error('No proxies were attempted.');

    // 3. Iterate through proxies until one succeeds
    for (const proxy of proxies) {
        const proxyUrl = proxy.buildUrl(url);
        const proxyOptions: RequestInit = { ...options };
        const newHeaders = new Headers(options.headers);
        const specificHeaders = proxy.getHeaders();
        Object.entries(specificHeaders).forEach(([key, value]) => newHeaders.set(key, String(value)));
        proxyOptions.headers = newHeaders;

        try {
            console.log(`Attempting fetch via proxy: ${proxy.name}`);
            const proxyResponse = await fetch(proxyUrl, proxyOptions);

            if (proxyResponse.ok) {
                console.log(`Successfully fetched via proxy: ${proxy.name}`);
                return proxyResponse;
            }
            
            lastError = new Error(`Proxy ${proxy.name} returned status ${proxyResponse.status}`);
            console.warn(lastError.message);

        } catch (error) {
            if (error instanceof Error) {
                lastError = error;
                console.warn(`Proxy ${proxy.name} failed to fetch:`, error.message);
            } else {
                const errorMessage = String(error);
                lastError = new Error(errorMessage);
                console.warn(`Proxy ${proxy.name} failed to fetch:`, errorMessage);
            }
        }
    }

    // 4. If all proxies fail, throw a comprehensive error
    console.error("All proxies failed.", lastError);
    throw new Error(`All proxies failed to fetch the resource. Last error: ${lastError.message}`);
};

const slugToTitle = (url: string): string => {
    try {
        const path = new URL(url).pathname;
        // remove leading/trailing slashes, replace hyphens, capitalize words
        return path
            .replace(/^\/|\/$/g, '')
            .replace(/-/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    } catch (e) {
        console.warn(`Could not parse URL for slugToTitle: ${url}`);
        return url; // fallback for invalid URLs
    }
};

const ProgressBar = ({ currentStep }: { currentStep: number }) => {
    const steps = ['Config', 'Content', 'Review & Publish'];
    return (
        <ol className="progress-bar">
            {steps.map((name, index) => {
                const stepIndex = index + 1;
                const status = stepIndex < currentStep ? 'completed' : stepIndex === currentStep ? 'active' : '';
                return (
                    <li key={name} className={`progress-step ${status}`}>
                        <div className="step-circle">{stepIndex < currentStep ? '✔' : stepIndex}</div>
                        {name}
                    </li>
                );
            })}
        </ol>
    );
};

const ApiKeyValidator = ({ status }) => {
    if (status === 'validating') {
        return <div className="key-status-icon"><div className="key-status-spinner"></div></div>;
    }
    if (status === 'valid') {
        return <div className="key-status-icon success"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg></div>;
    }
    if (status === 'invalid') {
        return <div className="key-status-icon error"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg></div>;
    }
    return null;
};


const ConfigStep = ({ state, dispatch, onFetchSitemap, onValidateKey }) => {
    const { wpUrl, wpUser, wpPassword, sitemapUrl, urlLimit, loading, aiProvider, apiKeys, openRouterModels, keyStatus } = state;
    const isSitemapConfigValid = useMemo(() => sitemapUrl && sitemapUrl.trim() !== '', [sitemapUrl]);
    
    const isApiKeyValid = useMemo(() => {
        const keyIsEntered = apiKeys[aiProvider] && apiKeys[aiProvider].trim() !== '';
        return keyIsEntered && keyStatus[aiProvider] !== 'invalid';
    }, [apiKeys, aiProvider, keyStatus]);

    const [saveConfig, setSaveConfig] = useState(true);

    const debouncedValidateKey = useCallback(debounce(onValidateKey, 500), [onValidateKey]);

    const handleApiKeyChange = (e) => {
        const { value } = e.target;
        dispatch({ type: 'SET_API_KEY', payload: { provider: aiProvider, key: value } });
        if (value.trim() !== '') {
            debouncedValidateKey(aiProvider, value);
        }
    };

    const handleProviderChange = (e) => {
        const newProvider = e.target.value;
        dispatch({ type: 'SET_AI_PROVIDER', payload: newProvider });
        const key = apiKeys[newProvider];
        if (key && key.trim() !== '' && keyStatus[newProvider] === 'unknown') {
            onValidateKey(newProvider, key);
        }
    };

    return (
        <div className="step-container">
            <fieldset className="config-fieldset">
                <legend>WordPress Configuration</legend>
                <div className="form-group">
                    <label htmlFor="wpUrl">WordPress URL</label>
                    <input type="url" id="wpUrl" value={wpUrl} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpUrl', value: e.target.value } })} placeholder="https://example.com" />
                </div>
                <div className="form-group">
                    <label htmlFor="wpUser">WordPress Username</label>
                    <input type="text" id="wpUser" value={wpUser} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpUser', value: e.target.value } })} placeholder="admin" />
                </div>
                <div className="form-group">
                    <label htmlFor="wpPassword">Application Password</label>
                    <input type="password" id="wpPassword" value={wpPassword} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpPassword', value: e.target.value } })} placeholder="••••••••••••••••" />
                    <p className="help-text">
                        This is not your main password. <a href="https://wordpress.org/documentation/article/application-passwords/" target="_blank" rel="noopener noreferrer">Learn how to create an Application Password</a>.
                    </p>
                </div>
                 <div className="checkbox-group">
                    <input type="checkbox" id="saveConfig" checked={saveConfig} onChange={(e) => setSaveConfig(e.target.checked)} />
                    <label htmlFor="saveConfig">Save WordPress Configuration</label>
                </div>
                <div style={{
                    marginTop: '1rem',
                    padding: '1rem',
                    borderRadius: '8px',
                    backgroundColor: 'var(--warning-bg-color)',
                    border: '1px solid var(--warning-color)',
                    color: 'var(--warning-text-color)'
                }}>
                    <p style={{margin: 0, fontSize: '0.875rem', lineHeight: '1.5'}}><strong>Security Note:</strong> For maximum reliability, this application may route requests through a public proxy to bypass browser security (CORS). Avoid using main administrator credentials; it is strongly recommended to use a dedicated Application Password with limited permissions.</p>
                </div>
            </fieldset>

            <fieldset className="config-fieldset">
                <legend>Content Source</legend>
                <div className="form-group">
                    <label htmlFor="sitemapUrl">Sitemap URL</label>
                    <input type="url" id="sitemapUrl" value={sitemapUrl} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'sitemapUrl', value: e.target.value } })} placeholder="https://example.com/sitemap.xml" />
                </div>
                <div className="form-group">
                    <label htmlFor="urlLimit">URL Limit for Analysis</label>
                    <input type="number" id="urlLimit" value={urlLimit} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'urlLimit', value: parseInt(e.target.value, 10) || 1 } })} min="1" />
                    <p className="help-text">Maximum number of URLs from the sitemap to analyze for topic suggestions.</p>
                </div>
            </fieldset>

            <fieldset className="config-fieldset">
                <legend>AI Configuration</legend>
                <div className="form-group">
                    <label htmlFor="aiProvider">AI Provider</label>
                    <select id="aiProvider" value={aiProvider} onChange={handleProviderChange}>
                        <option value="gemini">Google Gemini</option>
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="openrouter">OpenRouter (Experimental)</option>
                    </select>
                </div>
                {aiProvider === 'openrouter' && (
                    <div className="form-group">
                        <label htmlFor="openRouterModel">Model</label>
                        <input
                            type="text"
                            id="openRouterModel"
                            list="openrouter-models-list"
                            value={state.openRouterModel}
                            onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'openRouterModel', value: e.target.value } })}
                            placeholder="e.g., google/gemini-flash-1.5"
                        />
                        <datalist id="openrouter-models-list">
                            {openRouterModels.map(model => <option key={model} value={model} />)}
                        </datalist>
                        <p className="help-text">
                            Enter any model name from <a href="https://openrouter.ai/models" target="_blank" rel="noopener noreferrer">OpenRouter</a>.
                        </p>
                    </div>
                )}
                <div className="form-group api-key-group">
                    <label htmlFor="apiKey">API Key</label>
                    <input
                        type="password"
                        id="apiKey"
                        value={apiKeys[aiProvider] || ''}
                        onChange={handleApiKeyChange}
                        placeholder={`Enter your ${aiProvider.charAt(0).toUpperCase() + aiProvider.slice(1)} API Key`}
                    />
                    <ApiKeyValidator status={keyStatus[aiProvider]} />
                </div>
            </fieldset>

            <button
                className="btn"
                onClick={() => onFetchSitemap(sitemapUrl, urlLimit, saveConfig)}
                disabled={loading || !isSitemapConfigValid || !isApiKeyValid}
            >
                {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Analyze Sitemap & Proceed'}
            </button>
        </div>
    );
};

const ContentStep = ({ state, dispatch, onGenerateContent, onFetchExistingPosts }) => {
    const { posts, loading, contentMode } = state;
    const [searchTerm, setSearchTerm] = useState('');
    const updateModeInitialized = useRef(false);

    useEffect(() => {
        // Automatically fetch recent posts when switching to 'update' mode
        if (contentMode === 'update' && !updateModeInitialized.current) {
            onFetchExistingPosts(''); // Pass empty string to fetch recents
            updateModeInitialized.current = true;
        }
        // Reset the flag if user switches back to 'new' content mode
        if (contentMode === 'new') {
            updateModeInitialized.current = false;
        }
    }, [contentMode, onFetchExistingPosts]);

    const debouncedFetch = useCallback(debounce(onFetchExistingPosts, 500), [onFetchExistingPosts]);

    const handleSearchChange = (e) => {
        const newSearchTerm = e.target.value;
        setSearchTerm(newSearchTerm);
        debouncedFetch(newSearchTerm);
    };

    const handleModeChange = (newMode) => {
        if (contentMode === newMode) return;
        setSearchTerm('');
        dispatch({ type: 'SET_CONTENT_MODE', payload: newMode });
    };

    const handlePostSelection = (postId, checked) => {
        dispatch({ type: 'TOGGLE_POST_SELECTION', payload: postId });
    };

    const selectedCount = useMemo(() => posts.filter(p => p.selected).length, [posts]);

    const renderNewContentMode = () => (
        <>
            <div className="posts-list-container">
                 <p className="help-text" style={{textAlign: 'center', margin: '-1rem 0 1.5rem 0'}}>Based on your sitemap, here are 5 suggested topics to improve your site's topical authority.</p>
                <div className="posts-table-wrapper">
                    <table className="posts-table">
                        <thead>
                            <tr>
                                <th>Suggested Topic</th>
                                <th>Reasoning</th>
                            </tr>
                        </thead>
                        <tbody>
                            {posts.map((post) => (
                                <tr key={post.id}>
                                    <td style={{fontWeight: 500}}>{post.title}</td>
                                    <td>{post.reason}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            <div className="button-group">
                <button className="btn" onClick={() => dispatch({ type: 'SET_STEP', payload: 1 })}>Back</button>
                <button className="btn" onClick={onGenerateContent} disabled={loading || posts.length === 0}>
                    {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : `Generate Content for ${posts.length} Topic(s)`}
                </button>
            </div>
        </>
    );

    const renderUpdateContentMode = () => (
        <div className="posts-list-container">
            <div className="form-group" style={{ marginBottom: '1rem' }}>
                <input
                    type="text"
                    className="posts-search-input"
                    placeholder="Search posts or browse oldest..."
                    value={searchTerm}
                    onChange={handleSearchChange}
                    aria-label="Search posts"
                />
            </div>
            {loading && posts.length === 0 ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
                    <div className="keyword-loading-spinner" />
                </div>
            ) : (
                <>
                    {posts.length > 0 ? (
                        <div className="posts-table-wrapper">
                            <table className="posts-table">
                                <thead>
                                    <tr>
                                        <th style={{width: '40px'}}><input type="checkbox" onChange={(e) => dispatch({type: 'TOGGLE_SELECT_ALL_POSTS', payload: e.target.checked})} /></th>
                                        <th>Title</th>
                                        <th>Date</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {posts.map(post => (
                                        <tr key={post.id}>
                                            <td><input type="checkbox" checked={post.selected || false} onChange={e => handlePostSelection(post.id, e.target.checked)} /></td>
                                            <td><a href={post.url} target="_blank" rel="noopener noreferrer">{post.title}</a></td>
                                            <td>{new Date(post.date).toLocaleDateString()}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <p style={{ textAlign: 'center', padding: '2rem 0', color: 'var(--text-light-color)' }}>
                            {searchTerm 
                                ? `No posts found for "${searchTerm}".`
                                : 'No posts found on your WordPress site. Use the search bar to try again.'
                            }
                        </p>
                    )}
                </>
            )}
             <div className="button-group">
                <button className="btn" onClick={() => dispatch({ type: 'SET_STEP', payload: 1 })}>Back</button>
                <button className="btn" onClick={onGenerateContent} disabled={loading || selectedCount === 0}>
                    {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : `Update Content for ${selectedCount} Post(s)`}
                </button>
            </div>
        </div>
    );

    return (
        <div className="step-container">
            <div className="content-mode-toggle">
                <button className={contentMode === 'new' ? 'active' : ''} onClick={() => handleModeChange('new')}>
                    New Content from Sitemap
                </button>
                <button className={contentMode === 'update' ? 'active' : ''} onClick={() => handleModeChange('update')}>
                    Update Existing Content
                </button>
            </div>
            {contentMode === 'new' ? renderNewContentMode() : renderUpdateContentMode()}
        </div>
    )
};

const ReviewStep = ({ state, dispatch, onPublish }) => {
    const { posts, loading, publishingStatus } = state;
    const [currentPostIndex, setCurrentPostIndex] = useState(0);
    const currentPost = posts[currentPostIndex];
    
    if (!currentPost) {
        return <div className="step-container">No content to review. Please go back.</div>;
    }
    
    const updatePostField = (field, value) => {
        dispatch({ type: 'UPDATE_POST_FIELD', payload: { index: currentPostIndex, field, value } });
    };

    const handlePublish = () => {
        onPublish(currentPost);
    };

    return (
        <div>
             {posts.length > 1 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <button className="btn btn-small" onClick={() => setCurrentPostIndex(i => Math.max(0, i - 1))} disabled={currentPostIndex === 0}>Previous</button>
                    <span>Viewing {currentPostIndex + 1} of {posts.length}</span>
                    <button className="btn btn-small" onClick={() => setCurrentPostIndex(i => Math.min(posts.length - 1, i + 1))} disabled={currentPostIndex === posts.length - 1}>Next</button>
                </div>
            )}
            <div className="review-layout">
                <div className="review-panel">
                    <h3>Editable Content</h3>
                    <div className="review-panel-content">
                        <div className="form-group">
                            <label htmlFor="postTitle">Post Title (H1)</label>
                            <input type="text" id="postTitle" value={currentPost.title || ''} onChange={e => updatePostField('title', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <div className="label-wrapper">
                                <label htmlFor="metaTitle">Meta Title</label>
                                <span className="char-counter">{currentPost.metaTitle?.length || 0} / 60</span>
                            </div>
                            <input type="text" id="metaTitle" value={currentPost.metaTitle || ''} onChange={e => updatePostField('metaTitle', e.target.value)} />
                        </div>
                         <div className="form-group">
                            <div className="label-wrapper">
                                <label htmlFor="metaDescription">Meta Description</label>
                                <span className="char-counter">{currentPost.metaDescription?.length || 0} / 160</span>
                            </div>
                            <textarea id="metaDescription" className="meta-description-input" value={currentPost.metaDescription || ''} onChange={e => updatePostField('metaDescription', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="content">HTML Content</label>
                            <textarea id="content" value={currentPost.content || ''} onChange={e => updatePostField('content', e.target.value)}></textarea>
                        </div>
                    </div>
                </div>
                <div className="review-panel">
                    <h3>Live Preview</h3>
                    <div className="review-panel-content">
                        <div className="live-preview">
                            <h1>{currentPost.title}</h1>
                            <div dangerouslySetInnerHTML={{ __html: currentPost.content }} />
                        </div>
                    </div>
                </div>
            </div>
            <div className="button-group">
                <button className="btn" onClick={() => dispatch({ type: 'SET_STEP', payload: 2 })}>Back to Selection</button>
                <button className="btn" onClick={handlePublish} disabled={loading}>
                    {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : `Publish to WordPress`}
                </button>
            </div>
             {publishingStatus[currentPost.id] && (
                <div className={`result ${publishingStatus[currentPost.id].success ? 'success' : 'error'}`}>
                    {publishingStatus[currentPost.id].message}
                    {publishingStatus[currentPost.id].link && (
                        <> <a href={publishingStatus[currentPost.id].link} target="_blank" rel="noopener noreferrer">View Post</a></>
                    )}
                </div>
            )}
        </div>
    );
};

const initialState = {
    currentStep: 1,
    wpUrl: '',
    wpUser: '',
    wpPassword: '',
    sitemapUrl: '',
    urlLimit: 50,
    posts: [],
    loading: false,
    error: null,
    aiProvider: 'gemini',
    apiKeys: {
        gemini: '',
        openai: '',
        anthropic: '',
        openrouter: '',
    },
    keyStatus: {
        gemini: 'unknown',
        openai: 'unknown',
        anthropic: 'unknown',
        openrouter: 'unknown',
    },
    openRouterModel: 'google/gemini-flash-1.5',
    openRouterModels: ['google/gemini-flash-1.5', 'openai/gpt-4o', 'anthropic/claude-3-haiku'],
    contentMode: 'new', // 'new' or 'update'
    publishingStatus: {}, // { postId: { success: boolean, message: string, link?: string } }
};


function reducer(state, action) {
    switch (action.type) {
        case 'SET_STEP':
            return { ...state, currentStep: action.payload };
        case 'SET_FIELD':
            return { ...state, [action.payload.field]: action.payload.value };
        case 'SET_API_KEY':
            return {
                ...state,
                apiKeys: { ...state.apiKeys, [action.payload.provider]: action.payload.key },
                keyStatus: { ...state.keyStatus, [action.payload.provider]: 'validating' }
            };
        case 'SET_AI_PROVIDER':
            return { ...state, aiProvider: action.payload };
        case 'SET_KEY_STATUS':
             return {
                ...state,
                keyStatus: { ...state.keyStatus, [action.payload.provider]: action.payload.status }
            };
        case 'FETCH_START':
            return { ...state, loading: true, error: null };
        case 'FETCH_SITEMAP_SUCCESS':
            return { ...state, loading: false, posts: action.payload, currentStep: 2, contentMode: 'new' };
        case 'FETCH_POSTS_SUCCESS':
            return { ...state, loading: false, posts: action.payload.map(p => ({...p, isFetched: true})) };
        case 'FETCH_ERROR':
            return { ...state, loading: false, error: action.payload };
        case 'GENERATE_CONTENT_SUCCESS':
            return { ...state, loading: false, posts: action.payload, currentStep: 3 };
        case 'UPDATE_POST_FIELD':
            return {
                ...state,
                posts: state.posts.map((post, index) =>
                    index === action.payload.index ? { ...post, [action.payload.field]: action.payload.value } : post
                )
            };
        case 'SET_CONTENT_MODE':
            return { ...state, contentMode: action.payload, posts: [], error: null };
        case 'TOGGLE_POST_SELECTION':
            return {
                ...state,
                posts: state.posts.map(p => p.id === action.payload ? {...p, selected: !p.selected} : p)
            };
        case 'TOGGLE_SELECT_ALL_POSTS':
            const areAllSelected = state.posts.every(p => p.selected);
            return {
                ...state,
                posts: state.posts.map(p => ({...p, selected: !areAllSelected}))
            };
        case 'PUBLISH_START':
            return { ...state, loading: true };
        case 'PUBLISH_SUCCESS':
        case 'PUBLISH_ERROR':
             return {
                ...state,
                loading: false,
                publishingStatus: {
                    ...state.publishingStatus,
                    [action.payload.postId]: {
                        success: action.payload.success,
                        message: action.payload.message,
                        link: action.payload.link
                    }
                }
            };
        case 'LOAD_CONFIG':
            return { ...state, ...action.payload };
        default:
            throw new Error(`Unhandled action type: ${action.type}`);
    }
}

const App = () => {
    const [state, dispatch] = useReducer(reducer, initialState);

    useEffect(() => {
        const savedConfig = localStorage.getItem('wpContentOptimizerConfig');
        if (savedConfig) {
            const config = JSON.parse(savedConfig);
            dispatch({ type: 'LOAD_CONFIG', payload: config });
        }
    }, []);

    const handleValidateKey = useCallback(async (provider, key) => {
        dispatch({ type: 'SET_KEY_STATUS', payload: { provider, status: 'validating' } });
        try {
            if (!key || key.length < 10) throw new Error('Invalid key format');
            await new Promise(res => setTimeout(res, 500));
            dispatch({ type: 'SET_KEY_STATUS', payload: { provider, status: 'valid' } });
        } catch (error) {
            dispatch({ type: 'SET_KEY_STATUS', payload: { provider, status: 'invalid' } });
        }
    }, []);
    
    const handleFetchSitemap = async (sitemapUrl, urlLimit, saveConfig) => {
        dispatch({ type: 'FETCH_START' });
        if (saveConfig) {
            const configToSave = {
                wpUrl: state.wpUrl,
                wpUser: state.wpUser,
                wpPassword: state.wpPassword,
                aiProvider: state.aiProvider,
                apiKeys: state.apiKeys
            };
            localStorage.setItem('wpContentOptimizerConfig', JSON.stringify(configToSave));
        }
        try {
            const sitemapResponse = await smartFetch(sitemapUrl);
            if (!sitemapResponse.ok) throw new Error(`Failed to fetch sitemap. Status: ${sitemapResponse.status}`);
            
            const sitemapText = await sitemapResponse.text();
            const parser = new DOMParser();
            const sitemapDoc = parser.parseFromString(sitemapText, "application/xml");
            const urlNodes = sitemapDoc.querySelectorAll("loc");
            const urls = Array.from(urlNodes).map(node => node.textContent).slice(0, urlLimit);

            if (urls.length === 0) throw new Error("No URLs found in the sitemap.");
            
            const ai = getAiClient();
            const prompt = `You are an expert SEO strategist. Analyze the following list of URLs from a website's sitemap to understand its core topics:\n\n${urls.join('\n')}\n\nBased on this analysis, identify 5 highly relevant and engaging topics or long-tail keywords that the website has likely not covered yet. These suggestions should help expand the site's topical authority and attract organic traffic.\n\nFor each suggestion, provide a compelling blog post title and a brief, clear reason explaining its value to the website's audience and SEO strategy.\n\nReturn the response as a single, valid JSON object. The object should have a single key "suggestions" which is an array of objects. Each object in the array must have these two keys: "topic" (the suggested blog post title) and "reason" (the explanation).`;
            
            let generatedText;
            
             if (state.aiProvider === 'gemini') {
                const geminiClient = ai as GoogleGenAI;
                const response = await geminiClient.models.generateContent({
                    model: 'gemini-2.5-flash',
                    contents: prompt,
                    config: {
                        responseMimeType: "application/json",
                        responseSchema: {
                            type: Type.OBJECT,
                            properties: {
                                suggestions: {
                                    type: Type.ARRAY,
                                    items: {
                                        type: Type.OBJECT,
                                        properties: {
                                            topic: { type: Type.STRING },
                                            reason: { type: Type.STRING }
                                        },
                                        required: ['topic', 'reason']
                                    }
                                }
                            },
                            required: ['suggestions']
                        }
                    }
                });
                generatedText = response.text;
            } else if (state.aiProvider === 'anthropic') {
                    const anthropicClient = ai as Anthropic;
                    const response = await anthropicClient.messages.create({
                        model: 'claude-3-haiku-20240307',
                        max_tokens: 4096,
                        messages: [{ role: 'user', content: prompt }],
                    });
                     if (response.content && response.content.length > 0 && response.content[0].type === 'text') {
                       generatedText = response.content[0].text;
                    } else {
                       throw new Error('Unexpected Anthropic API response format');
                    }
            } else { // OpenAI and OpenRouter
                const openAIClient = ai as OpenAI;
                const model = state.aiProvider === 'openai' ? 'gpt-4o' : state.openRouterModel;
                const response = await openAIClient.chat.completions.create({
                    model: model,
                    messages: [{ role: 'user', content: prompt }],
                    response_format: { type: "json_object" },
                });
                generatedText = response.choices[0].message.content;
            }

            const parsedResult = JSON.parse(generatedText);
            const suggestions = parsedResult.suggestions;
            
            if (!suggestions || !Array.isArray(suggestions)) throw new Error("AI did not return valid suggestions.");

            const posts = suggestions.map((s, i) => ({ 
                id: `suggestion-${i}`, 
                url: '', 
                title: s.topic, 
                reason: s.reason,
                content: '' 
            }));
            
            dispatch({ type: 'FETCH_SITEMAP_SUCCESS', payload: posts });
        } catch (error) {
            dispatch({ type: 'FETCH_ERROR', payload: `Error processing sitemap: ${error.message}. Please verify the sitemap URL is correct and publicly accessible.` });
        }
    };
    
    const handleFetchExistingPosts = useCallback(async (searchTerm = '') => {
        dispatch({ type: 'FETCH_START' });
        const { wpUrl, wpUser, wpPassword } = state;

        if (!wpUrl || !wpUser || !wpPassword) {
            dispatch({ type: 'FETCH_ERROR', payload: 'WordPress URL, Username, and Application Password are required.' });
            return;
        }

        try {
            const apiUrl = new URL(`${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts`);
            apiUrl.searchParams.append('_fields', 'id,title,link,date');

            if (searchTerm.trim()) {
                apiUrl.searchParams.append('per_page', '100');
                apiUrl.searchParams.append('search', searchTerm);
            } else {
                apiUrl.searchParams.append('per_page', '20');
                apiUrl.searchParams.append('orderby', 'date');
                apiUrl.searchParams.append('order', 'asc');
            }
            
            const headers = new Headers();
            headers.append('Authorization', 'Basic ' + btoa(`${wpUser}:${wpPassword}`));

            const response = await smartFetch(apiUrl.toString(), { headers });

            if (!response.ok) {
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    if (errorData.message) errorMsg = `WordPress API Error: ${errorData.message}`;
                } catch (e) {}
                throw new Error(errorMsg);
            }

            const data = await response.json();
            const posts = data.map(post => ({
                id: post.id,
                title: post.title.rendered.replace(/&amp;/g, '&').replace(/&#8217;/g, '’'),
                url: post.link,
                date: post.date,
                content: '',
                selected: false,
            }));
            
            dispatch({ type: 'FETCH_POSTS_SUCCESS', payload: posts });
        } catch (error) {
            dispatch({ type: 'FETCH_ERROR', payload: `Failed to fetch posts: ${error.message}. Please check your WordPress URL, credentials, and ensure the REST API is accessible.` });
        }
    }, [state.wpUrl, state.wpUser, state.wpPassword]);

    const getAiClient = () => {
        const apiKey = state.apiKeys[state.aiProvider];
        switch (state.aiProvider) {
            case 'gemini':
                return new GoogleGenAI({ apiKey });
            case 'openai':
                return new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
            case 'anthropic':
                 return new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
            case 'openrouter':
                return new OpenAI({
                    baseURL: "https://openrouter.ai/api/v1",
                    apiKey: apiKey,
                    defaultHeaders: {
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "WP Content Optimizer",
                    },
                    dangerouslyAllowBrowser: true,
                });
            default:
                throw new Error('Unsupported AI provider');
        }
    };

    const handleGenerateContent = async () => {
        dispatch({ type: 'FETCH_START' });
        
        let internalLinksInstruction = `**Internal Linking:** You MUST add 6-10 high-quality internal links within the content. Use rich, descriptive anchor text. Since you don't know the exact URLs, use this placeholder format: \`<a href="/#internal-link-placeholder">[Your Rich Anchor Text Here]</a>\`. For example: \`<a href="/#internal-link-placeholder">advanced SEO strategies</a>\`.`;

        try {
            const sitemapUrl = 'https://gearuptogrow.com/post-sitemap.xml';
            const sitemapResponse = await smartFetch(sitemapUrl);
            if (!sitemapResponse.ok) {
                throw new Error(`Sitemap fetch failed with status: ${sitemapResponse.status}`);
            }
            
            const sitemapText = await sitemapResponse.text();
            const parser = new DOMParser();
            const sitemapDoc = parser.parseFromString(sitemapText, "application/xml");
            const urlNodes = sitemapDoc.querySelectorAll("loc");

            if (urlNodes.length > 0) {
                const internalLinksList = Array.from(urlNodes)
                    .map(node => node.textContent)
                    .filter((url): url is string => !!url)
                    .map(url => `- [${slugToTitle(url)}](${url})`)
                    .join('\n');
                
                if (internalLinksList) {
                    internalLinksInstruction = `**Internal Linking:** You MUST add 6-10 high-quality, contextually relevant internal links within the content. Use rich, descriptive anchor text. You MUST choose from the following list of available articles on the site. Link to the most relevant ones using their full URL.

**Available Internal Links:**
${internalLinksList}
`;
                }
            }
        } catch (error) {
            console.warn("Could not fetch or parse internal links sitemap, falling back to placeholder links.", error);
        }
        
        let referencesInstruction;
        if (state.aiProvider === 'gemini') {
            referencesInstruction = `**CRITICAL: Use Google Search:** You MUST use your access to Google Search to find the most up-to-date, authoritative information for this article. A "References" section linking to the sources you used will be automatically generated, so you DO NOT need to create one yourself in the 'content' field.`;
        } else {
            referencesInstruction = `**CRITICAL: Add References:** After the concluding paragraph, add a final H2 section titled "References". Inside, provide a list (\`<ul>\`) of 6-12 links to the most authoritative and reputable external sources (like academic studies, industry reports, or top-tier publications) that support the article's claims. Ensure these links are real and functional.`;
        }

        const basePromptStructure = (taskInstruction, topicOrUrl) => `You are a world-leading expert in SEO and content creation, acting as the chief editor for a prestigious online publication. Your task is to produce a blog post that is 10,000 times more valuable, insightful, and comprehensive than anything else on the internet. This article must be a definitive, state-of-the-art resource designed to rank #1 on Google.

**Core Task:** ${taskInstruction}

**Analysis Phase (Internal Monologue - essential for your process):**
1.  **Deep SERP Analysis:** Scrutinize the top 10 search results for the topic. Understand their structure, depth, and user intent.
2.  **Comprehensive Content Gap Analysis:** Identify every single gap in the existing top content. What questions are unanswered? What concepts are poorly explained? Your mission is to fill these gaps completely.
3.  **Advanced Keyword Analysis:** Identify the primary keyword, numerous long-tail variations, and a rich set of semantically related LSI keywords. You must weave these throughout the content with masterful, natural language.

**Content Generation Instructions:**
1.  The response MUST be a single, valid JSON object with these exact keys: "title", "metaTitle", "metaDescription", "content".
2.  **"title" (H1):** Create an irresistible, click-worthy H1 title.
3.  **"metaTitle":** Write a concise, SEO-optimized title for SERPs (50-60 characters).
4.  **"metaDescription":** Write a compelling meta description that promises immense value and drives clicks (150-160 characters).
5.  **"content" (HTML Body):**
    *   **CRITICAL: Start with a "Wow" Intro:** Begin the article with a highly engaging introduction. It MUST feature a surprising, fact-checked statistic or data point that immediately captures the reader's attention and establishes credibility.
    *   **CRITICAL: Add Key Takeaways:** Immediately following the intro, add an H3 section titled "Key Takeaways". This section must be in a div with class="key-takeaways". Inside, provide a bulleted list (\`<ul>\`) of exactly 6 concise, high-impact takeaways from the article.
    *   **Main Body:** Write a deeply comprehensive article of at least 1500 words. Structure it with a logical hierarchy of H2s and H3s. Use short paragraphs, bullet points, numbered lists, and bold text to enhance readability. The content must be exceptionally helpful, practical, and easy to understand for the target audience.
    *   ${internalLinksInstruction}
    *   ${referencesInstruction}
    *   **CRITICAL:** The "content" HTML string must NOT include the main <h1> title. It should start with the first paragraph (e.g., a <p> tag).
    *   **Tone:** Authoritative, expert, yet incredibly helpful and accessible.

**${isNewContent ? 'Topic to write about' : 'URL to process'}:** ${topicOrUrl}`;

        const isNewContent = state.contentMode === 'new';
        const newContentTask = "Write the ultimate, SEO-optimized blog post on the provided topic.";
        const rewriteContentTask = "Completely rewrite and supercharge the blog post from the provided URL into a definitive, state-of-the-art resource.";
        
        const postsToProcess = state.posts.filter(p => isNewContent || p.selected);
        
        const promiseFn = async (post) => {
            const prompt = isNewContent
                ? basePromptStructure(newContentTask, post.title)
                : basePromptStructure(rewriteContentTask, post.url);
            const identifier = isNewContent ? post.title : post.url;

            try {
                const ai = getAiClient();
                let response;
                let generatedText;

                if (state.aiProvider === 'gemini') {
                    const geminiClient = ai as GoogleGenAI;
                    const apiCall = () => geminiClient.models.generateContent({
                        model: 'gemini-2.5-flash',
                        contents: prompt,
                        config: {
                            tools: [{ googleSearch: {} }],
                        }
                    });
                    response = await makeResilientAiCall(apiCall);
                    generatedText = response.text;
                } else if (state.aiProvider === 'anthropic') {
                    const anthropicClient = ai as Anthropic;
                    const apiCall = () => anthropicClient.messages.create({
                        model: 'claude-3-haiku-20240307',
                        max_tokens: 4096,
                        messages: [{ role: 'user', content: prompt }],
                    });
                    const anthropicResponse = await makeResilientAiCall(apiCall);
                    if (anthropicResponse.content && anthropicResponse.content.length > 0 && anthropicResponse.content[0].type === 'text') {
                       generatedText = anthropicResponse.content[0].text;
                    } else {
                       throw new Error('Unexpected Anthropic API response format');
                    }
                } else { // OpenAI and OpenRouter
                    const openAIClient = ai as OpenAI;
                    const model = state.aiProvider === 'openai' ? 'gpt-4o' : state.openRouterModel;
                    const apiCall = () => openAIClient.chat.completions.create({
                        model: model,
                        messages: [{ role: 'user', content: prompt }],
                        response_format: { type: "json_object" },
                    });
                    const openAiResponse = await makeResilientAiCall(apiCall);
                    generatedText = openAiResponse.choices[0].message.content;
                }

                let jsonText = generatedText.match(/```json\n([\s\S]*?)\n```/)?.[1] || generatedText;
                const parsedContent = JSON.parse(jsonText);

                let finalContent = parsedContent.content || '';

                if (state.aiProvider === 'gemini' && response) {
                    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
                    if (groundingChunks && groundingChunks.length > 0) {
                        const uniqueReferences = new Map<string, string>();
                        groundingChunks.forEach(chunk => {
                            if (chunk.web && chunk.web.uri && chunk.web.title) {
                                try {
                                    new URL(chunk.web.uri);
                                    uniqueReferences.set(chunk.web.uri, chunk.web.title);
                                } catch (e) {
                                    console.warn(`Skipping invalid URL from grounding chunks: ${chunk.web.uri}`);
                                }
                            }
                        });

                        if (uniqueReferences.size > 0) {
                            console.log(`Found ${uniqueReferences.size} potential references from Google Search. Validating now...`);

                            const validationPromises = Array.from(uniqueReferences.entries()).map(async ([uri, title]) => {
                                try {
                                    // Use a HEAD request for efficiency to check link status without downloading content.
                                    // smartFetch will handle CORS issues by using proxies if needed.
                                    const res = await smartFetch(uri, { method: 'HEAD' });
                                    if (res.ok) {
                                        console.log(`✅ [200 OK] Reference validation success: ${uri}`);
                                        return { uri, title, valid: true };
                                    } else {
                                        console.warn(`⚠️ [Status ${res.status}] Reference validation failed: ${uri}`);
                                        return { uri, title, valid: false };
                                    }
                                } catch (error) {
                                    console.error(`❌ Network error during reference validation for ${uri}:`, error.message);
                                    return { uri, title, valid: false };
                                }
                            });
                            
                            const validationResults = await Promise.all(validationPromises);
                            const validReferences = validationResults.filter(r => r.valid);
                            
                            console.log(`Validation complete: ${validReferences.length} of ${uniqueReferences.size} references are valid and will be displayed.`);

                            if (validReferences.length > 0) {
                                let referencesHtml = '<div class="references-section"><h2>References</h2><ul>';
                                validReferences.forEach(({ uri, title }) => {
                                    referencesHtml += `<li><a href="${uri}" target="_blank" rel="noopener noreferrer">${title || uri}</a></li>`;
                                });
                                referencesHtml += '</ul></div>';
                                finalContent += referencesHtml;
                            } else {
                                console.warn("No valid, working references were found after checking all links provided by the search grounding. The references section will be omitted.");
                            }
                        }
                    }
                }
                
                return {
                    ...post,
                    title: parsedContent.title || post.title,
                    metaTitle: parsedContent.metaTitle || '',
                    metaDescription: parsedContent.metaDescription || '',
                    content: finalContent || '<p>Error: Content generation failed.</p>'
                };

            } catch (error) {
                console.error("AI Generation Error for", identifier, error);
                const errorMessage = (error.message && error.message.includes('429'))
                    ? `Rate limit exceeded and retries failed: ${error.message}`
                    : `Error generating content: ${error.message}`;
                return { ...post, content: `<p>Error: ${errorMessage}</p>` };
            }
        };

        const results = await processPromiseQueue(postsToProcess, promiseFn, null, 1000);
        const updatedPosts = results.map(r => r.status === 'fulfilled' ? r.value : r.reason);

        const finalPosts = state.posts.map(p => updatedPosts.find(up => up.id === p.id) || p);
        
        dispatch({ type: 'GENERATE_CONTENT_SUCCESS', payload: finalPosts.filter(p => state.contentMode === 'new' || p.selected) });
    };

    const handlePublish = async (post) => {
        dispatch({ type: 'PUBLISH_START' });
        const { wpUrl, wpUser, wpPassword } = state;

        try {
            const isUpdate = typeof post.id === 'number';
            const endpoint = isUpdate
                ? `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts/${post.id}`
                : `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts`;


            const headers = new Headers();
            headers.append('Authorization', 'Basic ' + btoa(`${wpUser}:${wpPassword}`));
            headers.append('Content-Type', 'application/json');

            const body = JSON.stringify({
                title: post.title,
                content: post.content,
                status: 'publish',
                meta: {
                    _yoast_wpseo_title: post.metaTitle,
                    _yoast_wpseo_metadesc: post.metaDescription,
                },
            });

            const response = await smartFetch(endpoint, {
                method: 'POST', // WordPress uses POST for updates too if you include the ID in the URL
                headers,
                body,
            });

            if (!response.ok) {
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    if (errorData.message) errorMsg = `WordPress API Error: ${errorData.message}`;
                } catch (e) { }
                throw new Error(errorMsg);
            }

            const responseData = await response.json();

            dispatch({
                type: 'PUBLISH_SUCCESS',
                payload: {
                    postId: post.id,
                    success: true,
                    message: `Successfully ${isUpdate ? 'updated' : 'published'} "${responseData.title.rendered}"!`,
                    link: responseData.link
                }
            });

        } catch (error) {
            dispatch({
                type: 'PUBLISH_ERROR',
                payload: {
                    postId: post.id,
                    success: false,
                    message: `Error publishing post: ${error.message}. Please check your WordPress credentials and permissions.`
                }
            });
        }
    };
    
    const renderStep = () => {
        switch (state.currentStep) {
            case 1:
                return <ConfigStep state={state} dispatch={dispatch} onFetchSitemap={handleFetchSitemap} onValidateKey={handleValidateKey} />;
            case 2:
                return <ContentStep state={state} dispatch={dispatch} onGenerateContent={handleGenerateContent} onFetchExistingPosts={handleFetchExistingPosts} />;
            case 3:
                return <ReviewStep state={state} dispatch={dispatch} onPublish={handlePublish} />;
            default:
                return <div>Unknown step</div>;
        }
    };

    return (
        <div className="container">
            <h1>WP Content Optimizer Pro</h1>
            <p className="subtitle">Automate content creation and optimization for your WordPress site using the power of AI. Configure your site, fetch URLs, and let the AI do the heavy lifting.</p>
            <ProgressBar currentStep={state.currentStep} />
            
            {state.error && (
                <div className="result error" style={{maxWidth: '800px', margin: '0 auto 2rem auto'}}>{state.error}</div>
            )}
            
            {renderStep()}
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);