/* ── ImageUploader: drag-drop + file input with instant preview ── */

import { useRef, useState, useCallback } from 'react';

interface Props {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function ImageUploader({ onFileSelect, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      setPreview(URL.createObjectURL(file));
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="image-uploader">
      <div
        className={`image-uploader__dropzone ${dragOver ? 'image-uploader__dropzone--active' : ''} ${disabled ? 'image-uploader__dropzone--disabled' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload ultrasound image"
      >
        {preview ? (
          <div className="image-uploader__preview-wrap">
            <img src={preview} alt="Uploaded ultrasound" className="image-uploader__preview" />
            <span className="image-uploader__change-hint">Click or drop to change</span>
          </div>
        ) : (
          <div className="image-uploader__placeholder">
            <svg className="image-uploader__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className="image-uploader__text">
              Drag &amp; drop an ultrasound image
            </p>
            <p className="image-uploader__subtext">or click to browse · JPEG / PNG</p>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png"
        onChange={onInputChange}
        className="image-uploader__input"
        disabled={disabled}
      />
    </div>
  );
}
