/* ── Disclaimer: static footer, always visible ── */

export default function Disclaimer() {
  return (
    <div className="disclaimer">
      <svg className="disclaimer__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="18" height="18">
        <path d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <p className="disclaimer__text">
        Research prototype — not a diagnostic device. Predictions and heatmaps are for model evaluation purposes only.
      </p>
    </div>
  );
}
