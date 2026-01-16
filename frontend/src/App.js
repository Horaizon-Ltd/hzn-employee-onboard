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

const UPLOAD_URL_FUNCTION_URL = process.env.REACT_APP_UPLOAD_URL;
const TRIGGER_JOB_URL = process.env.REACT_APP_TRIGGER_JOB_URL;
const CHECK_STATUS_URL = process.env.REACT_APP_CHECK_STATUS_URL;

const POLLING_INTERVAL_MS = 10000; // 10 seconds

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
    formats: '.xlsx, .xls',
    acceptedFormat: 'xlsx,xls'
  },
  {
    type: 'employee_holiday',
    label: 'Holiday (Feriepengeforpligtelse)',
    formats: '.xlsx, .xls',
    acceptedFormat: 'xlsx,xls'
  },
  {
    type: 'employee_general',
    label: 'Employee General (Medarbejderstamkort)',
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
  const [jobStatus, setJobStatus] = useState(null); // Current job status for display

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

  const downloadFile = async (downloadUrl, fileName) => {
    const fileResponse = await fetch(downloadUrl);
    const blob = await fileResponse.blob();

    const blobUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = fileName || 'danlon_processed_output.xlsx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    window.URL.revokeObjectURL(blobUrl);
  };

  const pollJobStatus = async (jobId) => {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const response = await fetch(`${CHECK_STATUS_URL}?job_id=${jobId}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            }
          });

          if (!response.ok) {
            throw new Error(`Failed to check status: ${response.status}`);
          }

          const data = await response.json();
          setJobStatus(data.status);

          if (data.status === 'DONE') {
            resolve(data);
          } else if (data.status === 'ERROR') {
            reject(new Error(data.error || 'Processing failed'));
          } else {
            // Still processing, poll again in 10 seconds
            setTimeout(poll, POLLING_INTERVAL_MS);
          }
        } catch (err) {
          reject(err);
        }
      };

      poll();
    });
  };

  const handleProcess = async () => {
    if (!selectedFiles.employee_active) {
      setError('Please upload the Active Employee List to start processing');
      return;
    }
    if (Object.keys(selectedFiles).length === 0) {
      setError('Please select at least one file to process');
      return;
    }

    setProcessing(true);
    setError(null);
    setSuccess(false);
    setJobStatus('UPLOADING');

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
      setJobStatus('UPLOADING');
      const uploadPromises = upload_urls.map(urlInfo => {
        const file = selectedFiles[urlInfo.file_type];
        return uploadFileToS3(file, urlInfo.upload_url, urlInfo.fields)
          .then(() => ({ file_type: urlInfo.file_type, s3_key: urlInfo.s3_key }));
      });

      const uploadedFiles = await Promise.all(uploadPromises);

      // Step 3: Trigger the processing job
      setJobStatus('STARTING');
      const triggerResponse = await fetch(TRIGGER_JOB_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`,
        },
        body: JSON.stringify({
          s3_files: uploadedFiles,
          session_id: session_id
        })
      });

      if (!triggerResponse.ok) {
        throw new Error(`Failed to start processing: ${triggerResponse.status}`);
      }

      const triggerData = await triggerResponse.json();
      const jobId = triggerData.job_id;

      if (!jobId) {
        throw new Error('No job ID received from server');
      }

      // Step 4: Poll for job completion
      setJobStatus('PENDING');
      const result = await pollJobStatus(jobId);

      // Step 5: Download the file
      if (result.download_url) {
        await downloadFile(result.download_url, result.file_name);
        setSuccess(true);
        setSelectedFiles({});
        setUploadKey(prev => prev + 1);
      } else {
        throw new Error('No download URL received');
      }

    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'An error occurred while processing files');
      setSelectedFiles({});
      setUploadKey(prev => prev + 1);
    } finally {
      setProcessing(false);
      setJobStatus(null);
    }
  };

  const selectedCount = Object.keys(selectedFiles).length;
  const hasActiveEmployee = Boolean(selectedFiles.employee_active);

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
              <p className="instruction">Choose the documents you want to process (Active Employee List required):</p>

              <div className="info-box">
                <strong>💡 Important:</strong> The Active Employee List is required to start processing.
                <ul>
                  <li><strong>Active Employee List (Medarbejderoversigt)</strong> - Required. Employee numbers from this file are used as the primary key.</li>
                  <li><strong>Employee General (Medarbejderstamkort)</strong> - Optional supplemental source for employee details.</li>
                  <li><strong>Payslip (Lønseddel)</strong> - Optional supplemental source for employee details.</li>
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
                  disabled={processing || !hasActiveEmployee}
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
                  <p>
                    {jobStatus === 'UPLOADING' && 'Uploading files to server...'}
                    {jobStatus === 'STARTING' && 'Starting processing job...'}
                    {jobStatus === 'PENDING' && 'Job queued, waiting to start...'}
                    {jobStatus === 'PROCESSING' && 'Processing documents (this may take 15-20 minutes for OCR)...'}
                    {!jobStatus && 'Processing...'}
                  </p>
                  <p className="processing-note">Please wait - your Excel file will download automatically when ready.</p>
                  {(jobStatus === 'PENDING' || jobStatus === 'PROCESSING') && (
                    <p className="processing-note">Checking status every 10 seconds...</p>
                  )}
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
