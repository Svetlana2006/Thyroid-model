/* ── DeviceBadge: shows CPU/GPU from /api/health ── */

import { useEffect, useState } from 'react';
import { getHealth } from '../api/client';

export default function DeviceBadge() {
  const [device, setDevice] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getHealth()
      .then((h) => setDevice(h.device))
      .catch(() => setError(true));
  }, []);

  if (error) {
    return (
      <span className="device-badge device-badge--offline" title="Backend unreachable">
        <span className="device-dot device-dot--offline" />
        Offline
      </span>
    );
  }

  if (!device) {
    return (
      <span className="device-badge device-badge--loading">
        <span className="device-dot device-dot--loading" />
        Connecting…
      </span>
    );
  }

  const isGpu = device.startsWith('cuda');
  return (
    <span className={`device-badge ${isGpu ? 'device-badge--gpu' : 'device-badge--cpu'}`}>
      <span className={`device-dot ${isGpu ? 'device-dot--gpu' : 'device-dot--cpu'}`} />
      {isGpu ? `GPU (${device})` : 'CPU'}
    </span>
  );
}
