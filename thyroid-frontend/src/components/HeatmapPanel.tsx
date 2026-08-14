/* ── HeatmapPanel: original vs overlay side-by-side per architecture ── */

import type { HeatmapEntry } from '../types';

interface Props {
  heatmaps: HeatmapEntry[];
  originalPreview: string | null;
}

const METHOD_LABELS: Record<string, string> = {
  'gradcam++': 'Grad-CAM++',
  'attention_rollout': 'Attention Rollout',
};

const ARCH_LABELS: Record<string, string> = {
  resnet50: 'ResNet-50',
  efficientnet_b3: 'EfficientNet-B3',
  swin_tiny: 'Swin-Tiny',
};

export default function HeatmapPanel({ heatmaps, originalPreview }: Props) {
  if (heatmaps.length === 0) return null;

  return (
    <div className="heatmap-panel">
      <h3 className="heatmap-panel__title">Explainability Heatmaps</h3>
      <div className="heatmap-panel__grid">
        {heatmaps.map((hm, i) => (
          <div key={i} className="heatmap-panel__card">
            <div className="heatmap-panel__card-header">
              <span className="heatmap-panel__arch">{ARCH_LABELS[hm.arch] ?? hm.arch}</span>
              <span className="heatmap-panel__method">{METHOD_LABELS[hm.method] ?? hm.method}</span>
            </div>
            <div className="heatmap-panel__images">
              {originalPreview && (
                <div className="heatmap-panel__img-wrap">
                  <img src={originalPreview} alt="Original" className="heatmap-panel__img" />
                  <span className="heatmap-panel__img-label">Original</span>
                </div>
              )}
              <div className="heatmap-panel__img-wrap">
                <img
                  src={`data:image/png;base64,${hm.image_base64}`}
                  alt={`${hm.method} heatmap for ${hm.arch}`}
                  className="heatmap-panel__img"
                />
                <span className="heatmap-panel__img-label">Heatmap Overlay</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
