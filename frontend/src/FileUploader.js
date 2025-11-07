import React from 'react';

const FileUploader = ({ fileType, label, acceptedFormats, onFileSelect, selectedFile }) => {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileSelect(fileType, file);
    }
  };

  return (
    <div className="file-uploader">
      <label className="file-label">
        <span className="label-text">{label}</span>
        <span className="format-hint">({acceptedFormats})</span>
      </label>
      <div className="file-input-wrapper">
        <input
          type="file"
          accept={acceptedFormats}
          onChange={handleFileChange}
          className="file-input"
          id={`file-${fileType}`}
        />
        {selectedFile && (
          <div className="file-selected">
            ✓ {selectedFile.name}
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploader;
