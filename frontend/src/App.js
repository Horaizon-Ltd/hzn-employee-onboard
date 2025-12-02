import React, { useState } from 'react';
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import { fetchAuthSession } from 'aws-amplify/auth';
import '@aws-amplify/ui-react/styles.css';
import FileUploader from './FileUploader';
import './App.css';
import awsConfig from './aws-config';

// Configure Amplify
Amplify.configure(awsConfig);

const LAMBDA_FUNCTION_URL = process.env.REACT_APP_LAMBDA_URL;
const UPLOAD_URL_FUNCTION_URL = process.env.REACT_APP_UPLOAD_URL;

const FILE_TYPES = [
  {
    type: 'employee_payslip',
    label: 'Payslip (Lønseddel)',
    formats: '.pdf',
    acceptedFormat: 'pdf'
  },
  {
    type: 'employee_active',
    label: 'Active Employee List (Medarbejderoversigt)',
    formats: '.xlsx',
    acceptedFormat: 'xlsx'
  },
  {
    type: 'employee_holiday',
    label: 'Holiday (Feriepengeforpligtelse)',
    formats: '.xlsx',
    acceptedFormat: 'xlsx'
  },
  {
    type: 'employee_general',
    label: 'Employee General (Medarbejder Stamkort)',
    formats: '.pdf',
    acceptedFormat: 'pdf'
  },
  {
    type: 'employee_list',
    label: 'Employee List (Medarbejderliste)',
    formats: '.pdf',
    acceptedFormat: 'pdf'
  }
];

function App() {
  const [selectedFiles, setSelectedFiles] = useState({});
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [uploadKey, setUploadKey] = useState(0); // Key to force re-render of file inputs

  const handleFileSelect = (fileType, file) => {
    setSelectedFiles(prev => ({
      ...prev,
      [fileType]: file
    }));
    setError(null);
    setSuccess(false);
  };

  const uploadFileToS3 = async (file, uploadUrl, fields) => {
    const formData = new FormData();

    // Add all fields from presigned POST
    Object.entries(fields).forEach(([key, value]) => {
      formData.append(key, value);
    });

    // Add the file last
    formData.append('file', file);

    const response = await fetch(uploadUrl, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`S3 upload failed: ${response.status}`);
    }

    return response;
  };

  const handleProcess = async () => {
    if (Object.keys(selectedFiles).length === 0) {
      setError('Please select at least one file to process');
      return;
    }

    setProcessing(true);
    setError(null);
    setSuccess(false);

    try {
      // Get JWT token from Amplify
      const session = await fetchAuthSession();
      const idToken = session.tokens?.idToken?.toString();

      if (!idToken) {
        throw new Error('Not authenticated. Please login again.');
      }

      // Step 1: Get presigned upload URLs
      const filesMetadata = Object.entries(selectedFiles).map(([fileType, file]) => ({
        file_type: fileType,
        file_name: file.name
      }));

      const uploadUrlsResponse = await fetch(UPLOAD_URL_FUNCTION_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          files: filesMetadata
        })
      });

      if (!uploadUrlsResponse.ok) {
        throw new Error(`Failed to get upload URLs: ${uploadUrlsResponse.status}`);
      }

      const uploadUrlsData = await uploadUrlsResponse.json();
      const { session_id, upload_urls } = uploadUrlsData;

      // Step 2: Upload all files to S3 in parallel
      const uploadPromises = upload_urls.map(urlInfo => {
        const file = selectedFiles[urlInfo.file_type];
        return uploadFileToS3(file, urlInfo.upload_url, urlInfo.fields)
          .then(() => ({ file_type: urlInfo.file_type, s3_key: urlInfo.s3_key }));
      });

      const uploadedFiles = await Promise.all(uploadPromises);

      // Step 3: Call processing Lambda with S3 keys
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15 * 60 * 1000); // 15 minutes

      const response = await fetch(LAMBDA_FUNCTION_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`,
        },
        body: JSON.stringify({
          s3_files: uploadedFiles,
          session_id: session_id
        }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        // Download the CSV from S3 presigned URL
        if (result.download_url) {
          // Fetch the file from S3 and trigger download
          const fileResponse = await fetch(result.download_url);
          const blob = await fileResponse.blob();

          // Create blob URL and trigger download
          const blobUrl = window.URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = blobUrl;
          link.download = result.file_name || 'danlon_processed_output.xlsx';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          // Clean up blob URL
          window.URL.revokeObjectURL(blobUrl);

          setSuccess(true);
          setSelectedFiles({});
          setUploadKey(prev => prev + 1); // Reset file inputs
        } else {
          throw new Error('No download URL received from server');
        }
      } else {
        throw new Error(result.error || 'Processing failed');
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'An error occurred while processing files');
      setSelectedFiles({}); // Clear files on error so user must reupload
      setUploadKey(prev => prev + 1); // Reset file inputs
    } finally {
      setProcessing(false);
    }
  };

  const selectedCount = Object.keys(selectedFiles).length;

  return (
    <Authenticator
      hideSignUp={true}
      loginMechanisms={['email']}
    >
      {({ signOut, user }) => (
        <div className="App">
          <header className="App-header">
            <div className="header-content">
              <div>
                <h1>Employee Onboarding Document Processor</h1>
                <p className="subtitle">Upload employee documents to generate Danlon Excel</p>
              </div>
              <div className="user-info">
                <span className="user-email">👤 {user?.signInDetails?.loginId || user?.username}</span>
                <button onClick={signOut} className="signout-button">Sign Out</button>
              </div>
            </div>
          </header>

          <main className="App-main">
            <div className="upload-section">
              <h2>Select Documents</h2>
              <p className="instruction">Choose the documents you want to process (at least one required):</p>

              <div className="info-box">
                <strong>💡 Important:</strong> Upload at least one of these files:
                <ul>
                  <li><strong>Active Employee List (Medarbejderoversigt)</strong> - Recommended as priority. Employee numbers from this file will be used as the primary key.</li>
                  <li><strong>Employee General (Medarbejder Stamkort)</strong> - Alternative source for employee numbers if Active List is not provided.</li>
                  <li><strong>Payslip (Lønseddel)</strong> - Can also be used as employee number source if the above are not available.</li>
                </ul>
              </div>

              <div className="file-uploaders">
                {FILE_TYPES.map(fileType => (
                  <FileUploader
                    key={`${fileType.type}-${uploadKey}`}
                    fileType={fileType.type}
                    label={fileType.label}
                    acceptedFormats={fileType.formats}
                    onFileSelect={handleFileSelect}
                    selectedFile={selectedFiles[fileType.type]}
                  />
                ))}
              </div>

              <div className="action-section">
                <div className="file-count">
                  {selectedCount} file{selectedCount !== 1 ? 's' : ''} selected
                </div>

                <button
                  onClick={handleProcess}
                  disabled={processing || selectedCount === 0}
                  className="process-button"
                >
                  {processing ? 'Processing...' : 'Process Documents'}
                </button>
              </div>

              {error && (
                <div className="message error-message">
                  ❌ {error}
                </div>
              )}

              {success && (
                <div className="message success-message">
                  ✅ Processing complete! Excel file downloaded.
                </div>
              )}

              {processing && (
                <div className="processing-info">
                  <div className="spinner"></div>
                  <p>Processing your documents... This may take 5-10 minutes for OCR processing.</p>
                  <p className="processing-note">Please wait - your CSV will download automatically when ready.</p>
                </div>
              )}
            </div>
          </main>

          <footer className="App-footer">
            <p>Powered by Horaizon</p>
          </footer>
        </div>
      )}
    </Authenticator>
  );
}

export default App;
