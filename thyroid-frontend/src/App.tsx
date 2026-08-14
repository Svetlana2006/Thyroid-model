/* ── App.tsx: single-page thyroid classification interface ── */

import { useEffect, useState } from 'react';
import { getModels, predict } from './api/client';
import type { ModelInfo, PredictResponse } from './types';
import DeviceBadge from './components/DeviceBadge';
import ModelSelector from './components/ModelSelector';
import ImageUploader from './components/ImageUploader';
import ResultCard from './components/ResultCard';
import HeatmapPanel from './components/HeatmapPanel';
import DetailsAccordion from './components/DetailsAccordion';
import Disclaimer from './components/Disclaimer';

type Status = 'idle' | 'loading' | 'error';

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [errorMsg, setErrorMsg] = useState<string>('');

  // Fetch models on mount
  useEffect(() => {
    getModels()
      .then((r) => {
        setModels(r.models);
        // Auto-select the full ensemble if available
        const fullEnsemble = r.models.find((m) => m.type === 'full_ensemble');
        if (fullEnsemble) setSelectedModelId(fullEnsemble.id);
        else if (r.models.length > 0) setSelectedModelId(r.models[0].id);
      })
      .catch(() => {
        /* models will be empty → ModelSelector shows disabled state */
      });
  }, []);

  const handleFileSelect = (file: File) => {
    setUploadedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setStatus('idle');
    setErrorMsg('');
  };

  const handleSubmit = async () => {
    if (!uploadedFile || !selectedModelId) return;
    setStatus('loading');
    setResult(null);
    setErrorMsg('');
    try {
      const r = await predict(uploadedFile, selectedModelId);
      setResult(r);
      setStatus('idle');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Prediction failed';
      setErrorMsg(msg);
      setStatus('error');
    }
  };

  const canSubmit = uploadedFile !== null && selectedModelId !== null && status !== 'loading';

  return (
    <div className="app">
      {/* Header */}
      <header className="app__header">
        <div className="app__header-inner">
          <div className="app__brand">
            <div className="app__logo">
              <svg viewBox="0 0 32 32" fill="none" width="36" height="36">
                <rect width="32" height="32" rx="8" fill="url(#logo-grad)" />
                <path d="M16 7v18M10 12h12M12 17h8" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
                <defs>
                  <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32">
                    <stop stopColor="#6366f1" />
                    <stop offset="1" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div>
              <h1 className="app__title">Thyroid Nodule Classifier</h1>
              <p className="app__subtitle">Deep Learning Prediction &amp; Explainability</p>
            </div>
          </div>
          <DeviceBadge />
        </div>
      </header>

      <main className="app__main">
        {/* Controls */}
        <section className="app__controls">
          <div className="app__controls-grid">
            <ModelSelector
              models={models}
              selectedId={selectedModelId}
              onChange={setSelectedModelId}
              disabled={status === 'loading'}
            />
            <ImageUploader
              onFileSelect={handleFileSelect}
              disabled={status === 'loading'}
            />
          </div>

          <button
            className="app__submit"
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            {status === 'loading' ? (
              <span className="app__submit-loading">
                <span className="spinner" />
                Analyzing…
              </span>
            ) : (
              'Analyze Image'
            )}
          </button>
        </section>

        {/* Error banner */}
        {status === 'error' && errorMsg && (
          <div className="app__error">
            <svg viewBox="0 0 20 20" fill="currentColor" width="20" height="20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Loading state */}
        {status === 'loading' && (
          <div className="app__loading-card">
            <div className="app__loading-pulse" />
            <p>Running inference — this may take a moment on CPU…</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="app__results">
            <ResultCard result={result} />
            <HeatmapPanel heatmaps={result.heatmaps} originalPreview={preview} />
            <DetailsAccordion result={result} />
          </section>
        )}

        <Disclaimer />
      </main>
    </div>
  );
}

export default App;
