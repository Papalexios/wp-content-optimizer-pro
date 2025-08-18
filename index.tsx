
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

const ContentCard = ({ post, onGenerate, onReview, generationStatus, isSelected, onToggleSelection, contentMode }) => {
    const isGenerated = generationStatus === 'done';
    const isGenerating = generationStatus === 'generating';

    return (
        <div className={`content-card ${isSelected ? 'selected' : ''} ${isGenerated ? 'generated' : ''}`}>
            <div className="content-card-header">
                <input
                    type="checkbox"
                    className="card-checkbox"
                    checked={isSelected}
                    onChange={onToggleSelection}
                    aria-label={`Select post: ${post.title}`}
                />
                <div className="header-main">
                    <h3>{post.title}</h3>
                    {isGenerated && <span className="status-badge generated">✔ Generated</span>}
                    {contentMode === 'update' && post.modified && !isGenerated && (
                        <span className="post-date">
                            Updated: {new Date(post.modified).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
                        </span>
                    )}
                </div>
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
    const { posts, loading, contentMode, generationStatus, selectedPostIds } = state;
    const selectedCount = selectedPostIds.size;
    
    const pendingPosts = posts.filter(p => generationStatus[p.id] !== 'done');
    const generatedPosts = posts.filter(p => generationStatus[p.id] === 'done');

    const isGenerateAllDisabled = loading || selectedCount === 0;

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
            
            {selectedCount > 0 && (
                <div className="selection-toolbar">
                    <span>{selectedCount} post{selectedCount !== 1 ? 's' : ''} selected</span>
                    <div className="selection-toolbar-actions">
                        {contentMode === 'update' && <button className="btn btn-secondary btn-small" onClick={() => dispatch({ type: 'SELECT_PENDING_POSTS' })}>Select Pending</button>}
                        <button className="btn btn-secondary btn-small" onClick={() => dispatch({ type: 'DESELECT_ALL' })}>Deselect All</button>
                    </div>
                </div>
            )}

            {contentMode === 'update' && posts.length === 0 && !loading && (
                 <div className="fetch-posts-prompt">
                    <p>Ready to update your existing content?</p>
                    <button className="btn" onClick={onFetchExistingPosts} disabled={loading}>
                        {loading ? <div className="spinner" style={{width: '24px', height: '24px', borderWidth: '2px'}}></div> : 'Fetch Recent Posts'}
                    </button>
                 </div>
            )}
            
            {contentMode === 'new' && posts.length > 0 && (
                <div className="content-cards-container">
                    {posts.map((post, postIndex) => (
                        <ContentCard 
                            key={post.id} 
                            post={post}
                            generationStatus={generationStatus[post.id]}
                            onGenerate={() => onGenerateContent(post)}
                            onReview={() => dispatch({ type: 'OPEN_REVIEW_MODAL', payload: postIndex })}
                            isSelected={selectedPostIds.has(post.id)}
                            onToggleSelection={() => dispatch({ type: 'TOGGLE_POST_SELECTION', payload: post.id })}
                            contentMode={contentMode}
                        />
                    ))}
                </div>
            )}

            {contentMode === 'update' && pendingPosts.length > 0 && (
                <>
                    <h2 className="content-group-header">Ready to Update</h2>
                    <div className="content-cards-container">
                        {pendingPosts.map((post, postIndex) => (
                             <ContentCard 
                                key={post.id} 
                                post={post}
                                generationStatus={generationStatus[post.id]}
                                onGenerate={() => onGenerateContent(post)}
                                onReview={() => dispatch({ type: 'OPEN_REVIEW_MODAL', payload: posts.findIndex(p => p.id === post.id) })}
                                isSelected={selectedPostIds.has(post.id)}
                                onToggleSelection={() => dispatch({ type: 'TOGGLE_POST_SELECTION', payload: post.id })}
                                contentMode={contentMode}
                            />
                        ))}
                    </div>
                </>
            )}

            {contentMode === 'update' && generatedPosts.length > 0 && (
                <>
                    <h2 className="content-group-header">Generated</h2>
                     <div className="content-cards-container">
                        {generatedPosts.map((post, postIndex) => (
                             <ContentCard 
                                key={post.id} 
                                post={post}
                                generationStatus={generationStatus[post.id]}
                                onGenerate={() => onGenerateContent(post)}
                                onReview={() => dispatch({ type: 'OPEN_REVIEW_MODAL', payload: posts.findIndex(p => p.id === post.id) })}
                                isSelected={selectedPostIds.has(post.id)}
                                onToggleSelection={() => dispatch({ type: 'TOGGLE_POST_SELECTION', payload: post.id })}
                                contentMode={contentMode}
                            />
                        ))}
                    </div>
                </>
            )}
            
            {posts.length > 0 && (
                <div className="button-group" style={{justifyContent: 'center'}}>
                    <button className="btn" onClick={onGenerateAll} disabled={isGenerateAllDisabled}>
                        {loading ? 'Generating...' : (selectedCount > 0 ? `Generate for ${selectedCount} Selected` : 'Generate All')}
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

const Footer = () => (
    <footer className="app-footer">
        <p>&copy; {new Date().getFullYear()} <a href="https://affiliatemarketingforsuccess.com" target="_blank" rel="noopener noreferrer">Affiliate Marketing For Success</a>. All Rights Reserved.</p>
        <p>Powered by the AI Content Engine</p>
    </footer>
);

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
    selectedPostIds: new Set(),
};

function reducer(state, action) {
    switch (action.type) {
        case 'SET_STEP': return { ...state, currentStep: action.payload };
        case 'SET_FIELD': return { ...state, [action.payload.field]: action.payload.value };
        case 'SET_API_KEY': return { ...state, apiKeys: { ...state.apiKeys, [action.payload.provider]: action.payload.key }, keyStatus: { ...state.keyStatus, [action.payload.provider]: 'validating' } };
        case 'SET_AI_PROVIDER': return { ...state, aiProvider: action.payload };
        case 'SET_KEY_STATUS': return { ...state, keyStatus: { ...state.keyStatus, [action.payload.provider]: action.payload.status } };
        case 'FETCH_START': return { ...state, loading: true, error: null };
        case 'FETCH_SITEMAP_SUCCESS': return { ...state, loading: false, posts: action.payload, currentStep: 2, contentMode: 'new', generationStatus: {}, selectedPostIds: new Set() };
        case 'FETCH_EXISTING_POSTS_SUCCESS': return { ...state, loading: false, posts: action.payload, generationStatus: {}, selectedPostIds: new Set() };
        case 'FETCH_ERROR': return { ...state, loading: false, error: action.payload };
        case 'SET_GENERATION_STATUS': return { ...state, generationStatus: { ...state.generationStatus, [action.payload.postId]: action.payload.status } };
        case 'GENERATE_SINGLE_POST_SUCCESS': return { ...state, posts: state.posts.map(p => p.id === action.payload.id ? action.payload : p) };
        case 'UPDATE_POST_FIELD': return { ...state, posts: state.posts.map((post, index) => index === action.payload.index ? { ...post, [action.payload.field]: action.payload.value } : post) };
        case 'SET_CONTENT_MODE': return { ...state, contentMode: action.payload, posts: [], error: null, generationStatus: {}, selectedPostIds: new Set() };
        case 'PUBLISH_START': return { ...state, loading: true };
        case 'PUBLISH_SUCCESS': case 'PUBLISH_ERROR': return { ...state, loading: false, publishingStatus: { ...state.publishingStatus, [action.payload.postId]: { success: action.payload.success, message: action.payload.message, link: action.payload.link } } };
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
        case 'SELECT_PENDING_POSTS': {
            const pendingPostIds = state.posts
                .filter(p => state.generationStatus[p.id] !== 'done')
                .map(p => p.id);
            return { ...state, selectedPostIds: new Set(pendingPostIds) };
        }
        case 'DESELECT_ALL': {
            return { ...state, selectedPostIds: new Set() };
        }
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

            const jsonText = generatedText.match(/```json\n([\s\S]*?)\n```/)?.[1] || generatedText;
            const suggestions = JSON.parse(jsonText).suggestions;
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
            const endpoint = `${wpUrl.replace(/\/$/, "")}/wp-json/wp/v2/posts?_fields=id,title,link,modified&per_page=20&orderby=modified&order=asc`;
            const headers = new Headers({ 'Authorization': 'Basic ' + btoa(`${wpUser}:${wpPassword}`) });
            const response = await smartFetch(endpoint, { headers });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
            }
            const existingPosts = await response.json();
            const formattedPosts = existingPosts.map(p => ({ id: p.id, title: p.title.rendered, url: p.link, modified: p.modified, content: '', reason: 'Existing post to be updated.' }));
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
            case 'openrouter': return new OpenAI({ baseURL: "https://openrouter.ai/api/v1", apiKey, defaultHeaders: { "HTTP-Referer": "http://localhost:3000", "X-Title": "AI Content Engine" }, dangerouslyAllowBrowser: true });
            default: throw new Error('Unsupported AI provider');
        }
    };

    const handleGenerateContent = async (postToProcess) => {
        dispatch({ type: 'SET_GENERATION_STATUS', payload: { postId: postToProcess.id, status: 'generating' } });

        const internalLinksList = PROMOTIONAL_LINKS.map(url => `- [${slugToTitle(url)}](${url})`).join('\n');
        const internalLinksInstruction = `**Internal Linking:** Your primary goal is to include 6-10 highly relevant internal links within the article body. You MUST choose them from the following list of high-value articles from affiliatemarketingforsuccess.com. Use rich, descriptive anchor text. Do NOT use placeholder links.\n\n**Available Internal Links:**\n${internalLinksList}`;
        
        const referencesInstruction = state.aiProvider === 'gemini' 
            ? `**CRITICAL: Use Google Search:** You MUST use Google Search for up-to-date, authoritative info. A "References" section will be auto-generated from real search results to ensure all links are valid and functional.`
            : `**CRITICAL: Add REAL, VERIFIABLE References:** After the conclusion, you MUST add an H2 section titled "References". In this section, provide a bulleted list (\`<ul>\`) of 6-12 links to REAL, CURRENT, and ACCESSIBLE authoritative external sources. Every link MUST be a fully functional, live URL that does not result in a 404 error. Do not invent or guess URLs. Your top priority is the accuracy and validity of these links.`;

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
        const postsToGenerate = state.selectedPostIds.size > 0
            ? state.posts.filter(p => state.selectedPostIds.has(p.id))
            : state.posts.filter(p => state.generationStatus[p.id] !== 'done');
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
                    <h1>AI Content Engine</h1>
                    <button className="btn btn-secondary btn-small" onClick={() => dispatch({type: 'SET_STEP', payload: 1})}>Edit Configuration</button>
                </header>
            )}
             {state.currentStep === 1 && (
                <>
                    <h1>AI Content Engine</h1>
                    <p className="subtitle">Your professional suite for creating and optimizing high-ranking WordPress content, powered by AI.</p>
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
            <Footer />
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
