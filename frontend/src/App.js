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
    label: 'Upload rapporten: Lønseddel (Payslip)',
    formats: 'pdf',
    acceptedFormat: 'pdf',
    hint: 'Findes under fanen LØN → Lønafregning → Lønsedler'
  },
  {
    type: 'employee_active',
    label: 'Upload rapporten: Medarbejderoversigt (List of active employees)',
    formats: '.xlsx, .xls',
    acceptedFormat: 'xlsx,xls',
    hint: 'Findes under fanen VIRKSOMHED → Udskrifter → Medarbejderoversigt. Der sorteres på aktive medarbejdere og der vælges "Vis som Excel" og gemmes.'
  },
  {
    type: 'employee_holiday',
    label: 'Upload rapporten: Feriepengeforpligtelse (Holiday pay obligation)',
    formats: '.xlsx, .xls',
    acceptedFormat: 'xlsx,xls',
    hint: 'Findes under fanen LØN → Ferie → Feriepengeforpligtigelse. Der vælges "Vis som Excel" og gemmes.'
  },
  {
    type: 'employee_general',
    label: 'Upload rapporten: Medarbejderstamkort (Employee masterdata)',
    formats: 'pdf',
    acceptedFormat: 'pdf',
    hint: 'Findes under fanen VIRKSOMHED → Udskrifter → Medarbejderstamkort'
  },
  {
    type: 'employee_list',
    label: 'Upload rapporten: Medarbejderliste (Employee List)',
    formats: 'pdf',
    acceptedFormat: 'pdf',
    hint: 'Findes under fanen VIRKSOMHED → Udskrifter → Medarbejderliste'
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
      setError('Upload venligst Medarbejderoversigt for at starte konverteringen');
      return;
    }
    if (Object.keys(selectedFiles).length === 0) {
      setError('Vælg venligst mindst én fil at konvertere');
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
      setError(err.message || 'Der opstod en fejl under behandlingen af filerne');
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
                <h1>Konverteringsværktøj. Fra Danløn til Zenegy®</h1>
                <p className="subtitle">Hurtig onboarding af medarbejdere</p>
              </div>
              <div className="user-info">
                <span className="user-email">👤 {user?.signInDetails?.loginId || user?.username}</span>
                <button onClick={signOut} className="signout-button">Log ud</button>
              </div>
            </div>
          </header>

          <main className="App-main">
            <div className="guide-section">
              <h2>Vejledning</h2>
              <p className="guide-intro">
                Dette værktøj konverterer rapporter fra Danløn til et format, der kan importeres direkte i Zenegy® Payroll.
                Følg nedenstående trin for at gennemføre konverteringen.
              </p>

              <div className="guide-step">
                <h3>Trin 1 – Download rapporter fra Danløn</h3>
                <p>Log ind i Danløn og download følgende rapporter:</p>
                <ul>
                  <li><strong>Medarbejderoversigt</strong> – Excel-fil (.xlsx)</li>
                  <li><strong>Medarbejder stamdata</strong> – PDF-fil (.pdf)</li>
                  <li><strong>Lønseddel</strong> – PDF-fil (.pdf)</li>
                  <li><strong>Skyldig feriepenge (Feriepengeforpligtelse)</strong> – Excel-fil (.xlsx)</li>
                  <li><strong>Medarbejderliste</strong> – PDF-fil (.pdf)</li>
                </ul>
              </div>

              <div className="guide-step">
                <h3>Trin 2 – Upload rapporterne til filkonverteren</h3>
                <p className="guide-note">
                  <strong>Vigtigt!</strong> Vi anbefaler at du uploader alle filerne, da det giver det mest komplette resultat, men du skal uploade mindst én af disse filer:
                </p>
                <ul>
                  <li>
                    <strong>Aktiv medarbejderliste (Medarbejderoversigt)</strong><br />
                    Anbefales som førstevalg. Medarbejdernumre fra denne fil bruges som primær nøgle.
                  </li>
                  <li>
                    <strong>Medarbejder generelt (Medarbejder Stamkort)</strong><br />
                    Alternativ kilde til medarbejdernumre, hvis Aktiv medarbejderliste mangler.
                  </li>
                  <li>
                    <strong>Lønseddel (Lønseddel)</strong><br />
                    Kan også bruges til medarbejdernumre, hvis de to ovenstående ikke er tilgængelige.
                  </li>
                </ul>
              </div>

              <div className="guide-step">
                <h3>Trin 3 – Klik på knappen "Konvertér dokumenter"</h3>
                <p>
                  Når behandlingen er færdig, konverteres alle relevante data og samles i én Excel-fil. Filen lander i din downloadmappe, klar til import i Zenegy.
                </p>
              </div>

              <div className="guide-step">
                <h3>Efter konverteringen – Importér filen i Payroll</h3>
                <ol>
                  <li>
                    Sørg for, at du har installeret Zenegy® Payroll filimporteren, du kan læse mere om filimporteren {' '}
                    <a href="https://help.zenegy.com/da/articles/760-installer-og-brug-zenegy-filimport" target="_blank" rel="noopener noreferrer">
                      her
                    </a>.
                  </li>
                  <li>
                    Upload filen i Zenegys filimporter under "Medarbejdere".
                  </li>
                </ol>
              </div>
            </div>

            <div className="upload-section">
              <h2>Upload og konverter filer</h2>

              <div className="file-uploaders">
                {FILE_TYPES.map(fileType => (
                  <FileUploader
                    key={`${fileType.type}-${uploadKey}`}
                    fileType={fileType.type}
                    label={fileType.label}
                    acceptedFormats={fileType.formats}
                    onFileSelect={handleFileSelect}
                    selectedFile={selectedFiles[fileType.type]}
                    hint={fileType.hint}
                  />
                ))}
              </div>

              <div className="action-section">
                <div className="file-count">
                  {selectedCount} fil{selectedCount !== 1 ? 'er' : ''} valgt
                </div>

                <button
                  onClick={handleProcess}
                  disabled={processing || !hasActiveEmployee}
                  className="process-button"
                >
                  {processing ? 'Konverterer...' : 'Konvertér dokumenter'}
                </button>
              </div>

              {error && (
                <div className="message error-message">
                  ❌ {error}
                </div>
              )}

              {success && (
                <div className="message success-message">
                  ✅ Konvertering fuldført! Excel-filen er downloadet.
                </div>
              )}

              {processing && (
                <div className="processing-info">
                  <div className="spinner"></div>
                  <p>
                    {jobStatus === 'UPLOADING' && 'Uploader filer til serveren...'}
                    {jobStatus === 'STARTING' && 'Starter konverteringsjob...'}
                    {jobStatus === 'PENDING' && 'Job i kø, venter på at starte...'}
                    {jobStatus === 'PROCESSING' && 'Behandler dokumenter (dette kan tage 15-20 minutter for OCR)...'}
                    {!jobStatus && 'Behandler...'}
                  </p>
                  <p className="processing-note">Vent venligst – din Excel-fil downloades automatisk, når den er klar.</p>
                  {(jobStatus === 'PENDING' || jobStatus === 'PROCESSING') && (
                    <p className="processing-note">Tjekker status hvert 10. sekund...</p>
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
