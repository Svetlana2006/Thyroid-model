/* ── ResultCard: headline prediction, probability, confidence, threshold ── */

import type { PredictResponse } from '../types';

interface Props {
  result: PredictResponse;
}

export default function ResultCard({ result }: Props) {
  const isMalignant = result.prediction === 'malignant';
  const pct = (result.probability * 100).toFixed(1);
  const isLow = result.confidence === 'low';

  return (
    <div className={`result-card ${isMalignant ? 'result-card--malignant' : 'result-card--benign'}`}>
      <div className="result-card__header">
        <span className={`result-card__class ${isMalignant ? 'result-card__class--malignant' : 'result-card__class--benign'}`}>
          {isMalignant ? 'Malignant' : 'Benign'}
        </span>
      </div>

      <div className="result-card__probability">
        <span className="result-card__probability-label">Malignant probability</span>
        <span className="result-card__probability-value">{pct}%</span>
        {!result.calibrated && (
          <span className="result-card__uncalibrated">
            Uncalibrated — run test-set evaluation for this model
          </span>
        )}
      </div>

      {/* Probability bar */}
      <div className="result-card__bar-container">
        <div className="result-card__bar-track">
          <div
            className={`result-card__bar-fill ${isMalignant ? 'result-card__bar-fill--malignant' : 'result-card__bar-fill--benign'}`}
            style={{ width: `${Math.min(result.probability * 100, 100)}%` }}
          />
          <div
            className="result-card__bar-threshold"
            style={{ left: `${result.threshold * 100}%` }}
            title={`Threshold: ${result.threshold.toFixed(2)}`}
          />
        </div>
        <div className="result-card__bar-labels">
          <span>0%</span>
          <span>100%</span>
        </div>
      </div>

      <div className={`result-card__confidence ${isLow ? 'result-card__confidence--low' : 'result-card__confidence--high'}`}>
        {isLow ? (
          <>⚠ Low confidence — recommend specialist review</>
        ) : (
          <>High confidence</>
        )}
      </div>

      <div className="result-card__threshold">
        Decision threshold: {result.threshold.toFixed(2)} (Youden's J)
      </div>
    </div>
  );
}
