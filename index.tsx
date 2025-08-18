
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
            const isRateLimitError = (
                (error.status === 429) || 
                (error.message && error.message.includes('429')) ||
                (error.message && error.message.toLowerCase().includes('rate limit'))
            );

            if (isRateLimitError && i < maxRetries - 1) {
                const delay = initialDelay * Math.pow(2, i) + Math.random() * 1000;
                console.warn(`Rate limit error detected. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${i + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                console.error(`AI call failed on attempt ${i + 1}.`, error);
                throw error;
            }
        }
    }
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

const slugToTitle = (url: string): string => {
    try {
        return new URL(url).pathname.replace(/^\/|\/$/g, '').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
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
            <fieldset className="config-fieldset">
                <legend>WordPress Configuration</legend>
                <div className="form-group"><label htmlFor="wpUrl">WordPress URL</label><input type="url" id="wpUrl" value={wpUrl} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpUrl', value: e.target.value } })} placeholder="https://example.com" /></div>
                <div className="form-group"><label htmlFor="wpUser">WordPress Username</label><input type="text" id="wpUser" value={wpUser} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpUser', value: e.target.value } })} placeholder="admin" /></div>
                <div className="form-group"><label htmlFor="wpPassword">Application Password</label><input type="password" id="wpPassword" value={wpPassword} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'wpPassword', value: e.target.value } })} placeholder="••••••••••••••••" /><p className="help-text">This is not your main password. <a href="https://wordpress.org/documentation/article/application-passwords/" target="_blank" rel="noopener noreferrer">Learn how to create one</a>.</p></div>
                <div className="checkbox-group"><input type="checkbox" id="saveConfig" checked={saveConfig} onChange={(e) => setSaveConfig(e.target.checked)} /><label htmlFor="saveConfig">Save WordPress Configuration</label></div>
                <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: '8px', backgroundColor: 'var(--warning-bg-color)', border: '1px solid var(--warning-color)', color: 'var(--warning-text-color)' }}><p style={{margin: 0, fontSize: '0.875rem', lineHeight: '1.5'}}><strong>Security Note:</strong> For reliability, this app may use public proxies to bypass browser security (CORS). Use a dedicated Application Password with limited permissions, not main admin credentials.</p></div>
            </fieldset>

            <fieldset className="config-fieldset">
                <legend>Content Source</legend>
                <div className="form-group"><label htmlFor="sitemapUrl">Sitemap URL</label><input type="url" id="sitemapUrl" value={sitemapUrl} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'sitemapUrl', value: e.target.value } })} placeholder="https://example.com/sitemap.xml" /></div>
                <div className="form-group"><label htmlFor="urlLimit">URL Limit for Analysis</label><input type="number" id="urlLimit" value={urlLimit} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'urlLimit', value: parseInt(e.target.value, 10) || 1 } })} min="1" /><p className="help-text">Max number of URLs from the sitemap to analyze for topic suggestions.</p></div>
            </fieldset>

            <fieldset className="config-fieldset">
                <legend>AI Configuration</legend>
                <div className="form-group"><label htmlFor="aiProvider">AI Provider</label><select id="aiProvider" value={aiProvider} onChange={handleProviderChange}><option value="gemini">Google Gemini</option><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="openrouter">OpenRouter (Experimental)</option></select></div>
                {aiProvider === 'openrouter' && (<div className="form-group"><label htmlFor="openRouterModel">Model</label><input type="text" id="openRouterModel" list="openrouter-models-list" value={state.openRouterModel} onChange={(e) => dispatch({ type: 'SET_FIELD', payload: { field: 'openRouterModel', value: e.target.value } })} placeholder="e.g., google/gemini-flash-1.5" /><datalist id="openrouter-models-list">{openRouterModels.map(model => <option key={model} value={model} />)}</datalist><p className="help-text">Enter any model name from <a href="https://openrouter.ai/models" target="_blank" rel="noopener noreferrer">OpenRouter</a>.</p></div>)}
                <div className="form-group api-key-group"><label htmlFor="apiKey">API Key</label><input type="password" id="apiKey" value={apiKeys[aiProvider] || ''} onChange={handleApiKeyChange} placeholder={`Enter your ${aiProvider.charAt(0).toUpperCase() + aiProvider.slice(1)} API Key`} /><ApiKeyValidator status={keyStatus[aiProvider]} /></div>
            </fieldset>

            <button className="btn" onClick={() => onFetchSitemap(sitemapUrl, urlLimit, saveConfig)} disabled={loading || !isSitemapConfigValid || !isApiKeyValid}>{loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Analyze Sitemap & Continue'}</button>
        </div>
    );
};

const ContentCard = ({ post, onGenerate, onReview, generationStatus }) => {
    const isGenerated = generationStatus === 'done';
    const isGenerating = generationStatus === 'generating';

    return (
        <div className="content-card">
            <div className="content-card-header">
                <h3>{post.title}</h3>
            </div>
            <div className="content-card-body">
                {post.reason && <p>{post.reason}</p>}
                {isGenerated && (
                     <textarea
                        className="generated-content-preview"
                        readOnly
                        value={post.content.substring(0, 200) + '...'}
                    />
                )}
            </div>
            <div className="content-card-actions">
                <button className="btn" onClick={isGenerated ? onReview : onGenerate} disabled={isGenerating}>
                    {isGenerating && <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div>}
                    {!isGenerating && (isGenerated ? 'Review & Publish' : 'Generate Content')}
                </button>
            </div>
        </div>
    );
};

const ContentStep = ({ state, dispatch, onGenerateContent, onFetchExistingPosts, onGenerateAll }) => {
    const { posts, loading, contentMode, generationStatus } = state;
    const isGenerateAllDisabled = loading || posts.length === 0 || posts.every(p => generationStatus[p.id] === 'done');

    return (
        <div className="step-container">
            <div className="content-mode-toggle">
                <button className={contentMode === 'new' ? 'active' : ''} onClick={() => dispatch({ type: 'SET_CONTENT_MODE', payload: 'new' })}>
                    New Content from Sitemap
                </button>
                <button className={contentMode === 'update' ? 'active' : ''} onClick={() => dispatch({ type: 'SET_CONTENT_MODE', payload: 'update' })}>
                    Update Existing Content
                </button>
            </div>
            
            {contentMode === 'update' && posts.length === 0 && !loading && (
                 <div className="fetch-posts-prompt">
                    <p>Ready to update your existing content?</p>
                    <button className="btn" onClick={onFetchExistingPosts} disabled={loading}>
                        {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Fetch Recent Posts'}
                    </button>
                 </div>
            )}
            
            {posts.length > 0 && (
                <div className="content-cards-container">
                    {posts.map((post, postIndex) => (
                        <ContentCard 
                            key={post.id} 
                            post={post}
                            generationStatus={generationStatus[post.id]}
                            onGenerate={() => onGenerateContent(post)}
                            onReview={() => dispatch({ type: 'OPEN_REVIEW_MODAL', payload: postIndex })}
                        />
                    ))}
                </div>
            )}
            
            {posts.length > 0 && (
                <div className="button-group" style={{justifyContent: 'center'}}>
                    <button className="btn" onClick={onGenerateAll} disabled={isGenerateAllDisabled}>
                        {loading ? 'Generating...' : 'Generate All'}
                    </button>
                </div>
            )}
        </div>
    );
};

const ReviewModal = ({ state, dispatch, onPublish, onClose }) => {
    const { posts, loading, publishingStatus, currentReviewIndex } = state;
    const [activeTab, setActiveTab] = useState('editor');
    const currentPost = posts[currentReviewIndex];
    
    useEffect(() => {
        // Reset to editor tab when post changes
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
                            <div className="form-group"><div className="label-wrapper"><label htmlFor="metaTitle">Meta Title</label><span className="char-counter">{currentPost.metaTitle?.length || 0} / 60</span></div><input type="text" id="metaTitle" value={currentPost.metaTitle || ''} onChange={e => updatePostField('metaTitle', e.target.value)} /></div>
                            <div className="form-group"><div className="label-wrapper"><label htmlFor="metaDescription">Meta Description</label><span className="char-counter">{currentPost.metaDescription?.length || 0} / 160</span></div><textarea id="metaDescription" className="meta-description-input" value={currentPost.metaDescription || ''} onChange={e => updatePostField('metaDescription', e.target.value)} /></div>
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
                 {publishingStatus[currentPost.id] && (
                    <div className={`result ${publishingStatus[currentPost.id].success ? 'success' : 'error'}`}>
                        {publishingStatus[currentPost.id].message}
                        {publishingStatus[currentPost.id].link && <>&nbsp;<a href={publishingStatus[currentPost.id].link} target="_blank" rel="noopener noreferrer">View Post</a></>}
                    </div>
                )}
            </div>
        </div>
    );
};

const initialState = {
    currentStep: 1,
    wpUrl: '', wpUser: '', wpPassword: '', sitemapUrl: '',
    urlLimit: 50,
    posts: [],
    loading: false, error: null,
    aiProvider: 'gemini',
    apiKeys: { gemini: '', openai: '', anthropic: '', openrouter: '' },
    keyStatus: { gemini: 'unknown', openai: 'unknown', anthropic: 'unknown', openrouter: 'unknown' },
    openRouterModel: 'google/gemini-flash-1.5',
    openRouterModels: ['google/gemini-flash-1.5', 'openai/gpt-4o', 'anthropic/claude-3-haiku'],
    contentMode: 'new',
    publishingStatus: {},
    generationStatus: {}, // { postId: 'idle' | 'generating' | 'done' | 'error' }
    currentReviewIndex: 0,
    isReviewModalOpen: false,
};

function reducer(state, action) {
    switch (action.type) {
        case 'SET_STEP': return { ...state, currentStep: action.payload };
        case 'SET_FIELD': return { ...state, [action.payload.field]: action.payload.value };
        case 'SET_API_KEY': return { ...state, apiKeys: { ...state.apiKeys, [action.payload.provider]: action.payload.key }, keyStatus: { ...state.keyStatus, [action.payload.provider]: 'validating' } };
        case 'SET_AI_PROVIDER': return { ...state, aiProvider: action.payload };
        case 'SET_KEY_STATUS': return { ...state, keyStatus: { ...state.keyStatus, [action.payload.provider]: action.payload.status } };
        case 'FETCH_START': return { ...state, loading: true, error: null };
        case 'FETCH_SITEMAP_SUCCESS': return { ...state, loading: false, posts: action.payload, currentStep: 2, contentMode: 'new', generationStatus: {} };
        case 'FETCH_EXISTING_POSTS_SUCCESS': return { ...state, loading: false, posts: action.payload, generationStatus: {} };
        case 'FETCH_ERROR': return { ...state, loading: false, error: action.payload };
        case 'SET_GENERATION_STATUS': return { ...state, generationStatus: { ...state.generationStatus, [action.payload.postId]: action.payload.status } };
        case 'GENERATE_SINGLE_POST_SUCCESS': return { ...state, posts: state.posts.map(p => p.id === action.payload.id ? action.payload : p) };
        case 'UPDATE_POST_FIELD': return { ...state, posts: state.posts.map((post, index) => index === action.payload.index ? { ...post, [action.payload.field]: action.payload.value } : post) };
        case 'SET_CONTENT_MODE': return { ...state, contentMode: action.payload, posts: [], error: null, generationStatus: {} };
        case 'PUBLISH_START': return { ...state, loading: true };
        case 'PUBLISH_SUCCESS': case 'PUBLISH_ERROR': return { ...state, loading: false, publishingStatus: { ...state.publishingStatus, [action.payload.postId]: { success: action.payload.success, message: action.payload.message, link: action.payload.link } } };
        case 'LOAD_CONFIG': return { ...state, ...action.payload };
        case 'SET_REVIEW_INDEX': return { ...state, currentReviewIndex: action.payload };
        case 'OPEN_REVIEW_MODAL': return { ...state, isReviewModalOpen: true, currentReviewIndex: action.payload };
        case 'CLOSE_REVIEW_MODAL': return { ...state, isReviewModalOpen: false };
        default: throw new Error(`Unhandled action type: ${action.type}`);
    }
}

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
    
    const handleFetchSitemap = async (sitemapUrl, urlLimit, saveConfig) => {
        dispatch({ type: 'FETCH_START' });
        if (saveConfig) localStorage.setItem('wpContentOptimizerConfig', JSON.stringify({ wpUrl: state.wpUrl, wpUser: state.wpUser, wpPassword: state.wpPassword, aiProvider: state.aiProvider, apiKeys: state.apiKeys }));
        
        try {
            const sitemapResponse = await smartFetch(sitemapUrl);
            if (!sitemapResponse.ok) throw new Error(`Failed to fetch sitemap. Status: ${sitemapResponse.status}`);
            
            const sitemapText = await sitemapResponse.text();
            const urls = Array.from(new DOMParser().parseFromString(sitemapText, "application/xml").querySelectorAll("loc")).map(node => node.textContent).slice(0, urlLimit);
            if (urls.length === 0) throw new Error("No URLs found in the sitemap.");
            
            const ai = getAiClient();
            const prompt = `You are an expert SEO strategist. Analyze these URLs: ${urls.join(', ')}. Identify 5 highly relevant, engaging topics or long-tail keywords the site likely hasn't covered to expand its topical authority. For each, provide a compelling blog post title and a brief reason for its value. Return a single, valid JSON object: { "suggestions": [ { "topic": "...", "reason": "..." } ] }`;
            
            let generatedText;
            if (state.aiProvider === 'gemini') {
                const response = await (ai as GoogleGenAI).models.generateContent({ model: 'gemini-2.5-flash', contents: prompt, config: { responseMimeType: "application/json", responseSchema: { type: Type.OBJECT, properties: { suggestions: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { topic: { type: Type.STRING }, reason: { type: Type.STRING } }, required: ['topic', 'reason'] } } }, required: ['suggestions'] } } });
                generatedText = response.text;
            } else if (state.aiProvider === 'anthropic') {
                const response = await (ai as Anthropic).messages.create({ model: 'claude-3-haiku-20240307', max_tokens: 4096, messages: [{ role: 'user', content: prompt }] });
                generatedText = response.content[0].type === 'text' ? response.content[0].text : '';
            } else { // OpenAI and OpenRouter
                const response = await (ai as OpenAI).chat.completions.create({ model: state.aiProvider === 'openai' ? 'gpt-4o' : state.openRouterModel, messages: [{ role: 'user', content: prompt }], response_format: { type: "json_object" } });
                generatedText = response.choices[0].message.content;
            }

            const suggestions = JSON.parse(generatedText).suggestions;
            if (!suggestions || !Array.isArray(suggestions)) throw new Error("AI did not return valid suggestions.");

            const posts = suggestions.map((s, i) => ({ id: `suggestion-${i}`, title: s.topic, reason: s.reason, content: '' }));
            dispatch({ type: 'FETCH_SITEMAP_SUCCESS', payload: posts });
        } catch (error) {
            dispatch({ type: 'FETCH_ERROR', payload: `Error processing sitemap: ${error.message}. Please verify the URL is correct and accessible.` });
        }
    };

    const handleFetchExistingPosts = async () => {
        dispatch({ type: 'FETCH_START' });
        const { wpUrl, wpUser, wpPassword } = state;
        try {
            const endpoint = `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts?_fields=id,title,link&per_page=20&orderby=date&order=desc`;
            const headers = new Headers({ 'Authorization': 'Basic ' + btoa(`${wpUser}:${wpPassword}`) });
            const response = await smartFetch(endpoint, { headers });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
            }
            const existingPosts = await response.json();
            const formattedPosts = existingPosts.map(p => ({ id: p.id, title: p.title.rendered, url: p.link, content: '', reason: 'Existing post to be updated.' }));
            dispatch({ type: 'FETCH_EXISTING_POSTS_SUCCESS', payload: formattedPosts });
        } catch (error) {
            dispatch({ type: 'FETCH_ERROR', payload: `Error fetching existing posts: ${error.message}` });
        }
    };
    
    const getAiClient = () => {
        const apiKey = state.apiKeys[state.aiProvider];
        switch (state.aiProvider) {
            case 'gemini': return new GoogleGenAI({ apiKey });
            case 'openai': return new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
            case 'anthropic': return new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
            case 'openrouter': return new OpenAI({ baseURL: "https://openrouter.ai/api/v1", apiKey, defaultHeaders: { "HTTP-Referer": "http://localhost:3000", "X-Title": "WP Content Optimizer" }, dangerouslyAllowBrowser: true });
            default: throw new Error('Unsupported AI provider');
        }
    };

    const handleGenerateContent = async (postToProcess) => {
        dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'generating' } });

        let internalLinksInstruction = `**Internal Linking:** Add 6-10 high-quality internal links using placeholder format: \`<a href="/#internal-link-placeholder">[Rich Anchor Text]</a>\`.`;
        try {
            const sitemapResponse = await smartFetch('https://gearuptogrow.com/post-sitemap.xml');
            if (sitemapResponse.ok) {
                const sitemapText = await sitemapResponse.text();
                const urlNodes = new DOMParser().parseFromString(sitemapText, "application/xml").querySelectorAll("loc");
                if (urlNodes.length > 0) {
                    const internalLinksList = Array.from(urlNodes).map(node => node.textContent).filter(Boolean).map(url => `- [${slugToTitle(url)}](${url})`).join('\n');
                    if (internalLinksList) internalLinksInstruction = `**Internal Linking:** Add 6-10 relevant internal links. Choose from this list:\n${internalLinksList}`;
                }
            }
        } catch (error) { console.warn("Could not fetch internal links sitemap, falling back to placeholders.", error); }
        
        const referencesInstruction = state.aiProvider === 'gemini' 
            ? `**CRITICAL: Use Google Search:** You MUST use Google Search for up-to-date, authoritative info. A "References" section will be auto-generated.`
            : `**CRITICAL: Add References:** After the conclusion, add an H2 section "References" with a list (\`<ul>\`) of 6-12 links to authoritative external sources.`;

        const isNewContent = state.contentMode === 'new';
        const task = isNewContent ? "Write the ultimate, SEO-optimized blog post on the topic." : "Completely rewrite and supercharge the blog post from the URL into a definitive resource.";
        const topicOrUrl = isNewContent ? postToProcess.title : postToProcess.url;
        
        const basePrompt = `You are a world-class SEO and content creator. Your task is to produce a blog post that is 10x better than anything online, designed to rank #1.
**Core Task:** ${task}
**Instructions:**
1.  Return a single, valid JSON object with keys: "title", "metaTitle" (50-60 chars), "metaDescription" (150-160 chars), "content" (HTML body).
2.  "content" MUST start with a "Wow" intro with a surprising statistic.
3.  Immediately after the intro, add an H3 "Key Takeaways" inside a \`<div class="key-takeaways">\` with a bulleted list of 6 takeaways.
4.  Write a comprehensive, 1500+ word article with H2s/H3s. Use short paragraphs, lists, and bolding.
5.  ${internalLinksInstruction}
6.  ${referencesInstruction}
7.  The "content" string MUST NOT include the main <h1> title.
**${isNewContent ? 'Topic' : 'URL'}:** ${topicOrUrl}`;

        try {
            const ai = getAiClient();
            let response, generatedText;

            if (state.aiProvider === 'gemini') {
                response = await makeResilientAiCall(() => (ai as GoogleGenAI).models.generateContent({ model: 'gemini-2.5-flash', contents: basePrompt, config: { tools: [{ googleSearch: {} }] } }));
                generatedText = response.text;
            } else if (state.aiProvider === 'anthropic') {
                const anthropicResponse = await makeResilientAiCall(() => (ai as Anthropic).messages.create({ model: 'claude-3-haiku-20240307', max_tokens: 4096, messages: [{ role: 'user', content: basePrompt }] }));
                generatedText = anthropicResponse.content[0].type === 'text' ? anthropicResponse.content[0].text : '';
            } else {
                const openAiResponse = await makeResilientAiCall(() => (ai as OpenAI).chat.completions.create({ model: state.aiProvider === 'openai' ? 'gpt-4o' : state.openRouterModel, messages: [{ role: 'user', content: basePrompt }], response_format: { type: "json_object" } }));
                generatedText = openAiResponse.choices[0].message.content;
            }

            const jsonText = generatedText.match(/```json\n([\s\S]*?)\n```/)?.[1] || generatedText;
            const parsedContent = JSON.parse(jsonText);
            let finalContent = parsedContent.content || '';

            if (state.aiProvider === 'gemini' && response?.candidates?.[0]?.groundingMetadata?.groundingChunks?.length > 0) {
                const uniqueReferences = new Map(response.candidates[0].groundingMetadata.groundingChunks.map(c => [c.web?.uri, c.web?.title]).filter(([uri]) => uri));
                if (uniqueReferences.size > 0) {
                    let referencesHtml = '<div class="references-section"><h2>References</h2><ul>';
                    uniqueReferences.forEach((title, uri) => {
                        referencesHtml += `<li><a href="${uri}" target="_blank" rel="noopener noreferrer">${title || uri}</a></li>`;
                    });
                    finalContent += referencesHtml + '</ul></div>';
                }
            }
            
            const updatedPost = { ...postToProcess, title: parsedContent.title || postToProcess.title, metaTitle: parsedContent.metaTitle || '', metaDescription: parsedContent.metaDescription || '', content: finalContent || '<p>Error: Content generation failed.</p>' };
            dispatch({ type: 'GENERATE_SINGLE_POST_SUCCESS', payload: updatedPost });
            dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'done' } });

        } catch (error) {
            console.error("AI Generation Error for", topicOrUrl, error);
            const errorMessage = (error.message && error.message.includes('429')) ? `Rate limit exceeded: ${error.message}` : `Error generating content: ${error.message}`;
            dispatch({ type: 'GENERATE_SINGLE_POST_SUCCESS', payload: { ...postToProcess, content: `<p>Error: ${errorMessage}</p>` } });
            dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'error' } });
        }
    };

    const handleGenerateAll = async () => {
        dispatch({type: 'FETCH_START'});
        const postsToGenerate = state.posts.filter(p => state.generationStatus[p.id] !== 'done');
        await processPromiseQueue(postsToGenerate, handleGenerateContent, null, 2000);
        dispatch({type: 'FETCH_ERROR', payload: null }); // to stop global loader
    };

    const handlePublish = async (post) => {
        dispatch({ type: 'PUBLISH_START' });
        const { wpUrl, wpUser, wpPassword } = state;
        try {
            const isUpdate = typeof post.id === 'number';
            const endpoint = isUpdate ? `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts/${post.id}` : `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts`;
            const headers = new Headers({ 'Authorization': 'Basic ' + btoa(`${wpUser}:${wpPassword}`), 'Content-Type': 'application/json' });
            const body = JSON.stringify({ title: post.title, content: post.content, status: 'publish', meta: { _yoast_wpseo_title: post.metaTitle, _yoast_wpseo_metadesc: post.metaDescription } });
            const response = await smartFetch(endpoint, { method: 'POST', headers, body });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
            }
            const responseData = await response.json();
            dispatch({ type: 'PUBLISH_SUCCESS', payload: { postId: post.id, success: true, message: `Successfully ${isUpdate ? 'updated' : 'published'} "${responseData.title.rendered}"!`, link: responseData.link } });
        } catch (error) {
            dispatch({ type: 'PUBLISH_ERROR', payload: { postId: post.id, success: false, message: `Error publishing post: ${error.message}` } });
        }
    };
    
    const renderContent = () => {
        switch (state.currentStep) {
            case 1: return <ConfigStep state={state} dispatch={dispatch} onFetchSitemap={handleFetchSitemap} onValidateKey={handleValidateKey} />;
            case 2: return <ContentStep state={state} dispatch={dispatch} onGenerateContent={handleGenerateContent} onGenerateAll={handleGenerateAll} onFetchExistingPosts={handleFetchExistingPosts} />;
            default: return <div>Unknown step</div>;
        }
    };

    return (
        <div className="container">
             {state.currentStep === 2 && (
                <header className="app-header">
                    <h1>Content Hub</h1>
                    <button className="btn btn-secondary btn-small" onClick={() => dispatch({type: 'SET_STEP', payload: 1})}>Edit Configuration</button>
                </header>
            )}
             {state.currentStep === 1 && (
                <>
                    <h1>WP Content Optimizer Pro</h1>
                    <p className="subtitle">Automate content creation and optimization for your WordPress site using the power of AI.</p>
                </>
            )}
            <ProgressBar currentStep={state.isReviewModalOpen ? 3 : state.currentStep} />
            {state.error && <div className="result error" style={{marginBottom: '2rem'}}>{state.error}</div>}
            {renderContent()}
            {state.isReviewModalOpen && (
                <ReviewModal 
                    state={state}
                    dispatch={dispatch}
                    onPublish={handlePublish}
                    onClose={() => dispatch({ type: 'CLOSE_REVIEW_MODAL' })}
                />
            )}
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
