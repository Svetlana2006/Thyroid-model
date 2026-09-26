/* ── DetailsAccordion: collapsed details — raw logit, temperature, per-model breakdown ── */

import { useState } from 'react';
import type { PredictResponse } from '../types';

interface Props {
  result: PredictResponse;
}

const ARCH_LABELS: Record<string, string> = {
  resnet50: 'ResNet-50',
  efficientnet_b3: 'EfficientNet-B3',
  swin_tiny: 'Swin-Tiny',
};

export default function DetailsAccordion({ result }: Props) {
  const [open, setOpen] = useState(false);

  const rawPct = (result.raw_probability * 100).toFixed(2);
  const calPct = (result.probability * 100).toFixed(2);

  return (
    <div className="details-accordion">
      <button
        className="details-accordion__toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="details-accordion__toggle-text">
          {open ? '▾' : '▸'} Technical Details
        </span>
      </button>

      {open && (
        <div className="details-accordion__body">
          {/* Raw stats */}
          <div className="details-accordion__section">
            <h4 className="details-accordion__section-title">Calibration</h4>
            <table className="details-accordion__table">
              <tbody>
                <tr>
                  <td className="details-accordion__cell-label">Raw (uncalibrated) probability</td>
                  <td className="details-accordion__cell-value">{rawPct}%</td>
                </tr>
                <tr>
                  <td className="details-accordion__cell-label">Temperature</td>
                  <td className="details-accordion__cell-value">{result.temperature.toFixed(4)}</td>
                </tr>
                <tr>
                  <td className="details-accordion__cell-label">Calibrated probability</td>
                  <td className="details-accordion__cell-value">{calPct}%</td>
                </tr>
                <tr>
                  <td className="details-accordion__cell-label">Uncertainty score</td>
                  <td className="details-accordion__cell-value">{result.uncertainty_score.toFixed(6)}</td>
                </tr>
                <tr>
                  <td className="details-accordion__cell-label">Device</td>
                  <td className="details-accordion__cell-value">{result.device.toUpperCase()}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Per-model breakdown (ensemble) */}
          {result.per_model_breakdown.length > 1 && (
            <div className="details-accordion__section">
              <h4 className="details-accordion__section-title">Per-Model Breakdown</h4>
              <table className="details-accordion__table details-accordion__table--full">
                <thead>
                  <tr>
                    <th>Architecture</th>
                    <th>Seed</th>
                    <th>Probability</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_model_breakdown.map((pm, i) => (
                    <tr key={i}>
                      <td>{ARCH_LABELS[pm.arch] ?? pm.arch}</td>
                      <td>{pm.seed}</td>
                      <td>{(pm.probability * 100).toFixed(2)}%</td>
                      <td>
                        {pm.weight !== null ? (
                          <span className={pm.weight === 0 ? 'details-accordion__zero-weight' : ''}>
                            {pm.weight.toFixed(2)}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Localization overlap */}
          <div className="details-accordion__section">
            {result.localization_overlap !== null ? (
              <p className="details-accordion__note">
                Localization overlap: {(result.localization_overlap * 100).toFixed(1)}%
              </p>
            ) : (
              <p className="details-accordion__note details-accordion__note--muted">
                No ground-truth region available — heatmap shown for reference only.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
