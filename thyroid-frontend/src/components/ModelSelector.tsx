/* ── ModelSelector: dropdown populated from GET /api/models ── */

import type { ModelInfo } from '../types';

interface Props {
  models: ModelInfo[];
  selectedId: string | null;
  onChange: (id: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({ models, selectedId, onChange, disabled }: Props) {
  if (models.length === 0) {
    return (
      <div className="model-selector model-selector--empty">
        <label className="model-selector__label">Model</label>
        <select disabled className="model-selector__select model-selector__select--disabled">
          <option>No trained models found</option>
        </select>
      </div>
    );
  }

  // Group by type
  const singles = models.filter((m) => m.type === 'single');
  const seedEnsembles = models.filter((m) => m.type === 'seed_ensemble');
  const fullEnsembles = models.filter((m) => m.type === 'full_ensemble');

  return (
    <div className="model-selector">
      <label className="model-selector__label" htmlFor="model-select">
        Select Model
      </label>
      <select
        id="model-select"
        className="model-selector__select"
        value={selectedId ?? ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        <option value="" disabled>
          Choose a model…
        </option>

        {fullEnsembles.length > 0 && (
          <optgroup label="Full Ensemble">
            {fullEnsembles.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </optgroup>
        )}

        {seedEnsembles.length > 0 && (
          <optgroup label="Seed Ensembles">
            {seedEnsembles.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </optgroup>
        )}

        {singles.length > 0 && (
          <optgroup label="Individual Models">
            {singles.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </div>
  );
}
