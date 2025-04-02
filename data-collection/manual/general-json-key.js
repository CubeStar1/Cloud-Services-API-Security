const fs = require('fs');
const path = require('path');

// Path to the logs directory and log file
const logDir = path.join(__dirname, 'logs');
const logFile = path.join(logDir, 'netlify-traffic-logs.json');

// Ensure the logs directory exists
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir);

// Helper function to log data to the file
function logToFile(content) {
  const logStream = fs.createWriteStream(logFile, { flags: 'a' });
  logStream.write(JSON.stringify(content) + ',\n');
  logStream.end();
}

module.exports = {

  // Intercept and log request details
  *beforeSendRequest(requestDetail) {
    const request = {
      type: 'request',
      url: requestDetail.url,
      method: requestDetail.requestOptions?.method || 'UNKNOWN',
      headers_Host: requestDetail.requestOptions.headers['Host'] || '',
      requestHeaders_Origin: requestDetail.requestOptions.headers['Origin'] || '',
      requestHeaders_Content_Type: requestDetail.requestOptions.headers['Content-Type'] || '',
      requestHeaders_Referer: requestDetail.requestOptions.headers['Referer'] || '',
      requestHeaders_Accept: requestDetail.requestOptions.headers['Accept'] || '',
      body: requestDetail.requestData ? requestDetail.requestData.toString() : null
    };

    logToFile(request);
    return null;
  },

  // Intercept and log response details
  *beforeSendResponse(requestDetail, responseDetail) {
    const response = {
      type: 'response',
      url: requestDetail.url,
      method: requestDetail.requestOptions?.method || 'UNKNOWN',
      headers_Host: requestDetail.url,
      responseHeaders_Content_Type: responseDetail.response.header['Content-Type'] || '',
      body: responseDetail.response.body ? responseDetail.response.body.toString() : null
    };

    logToFile(response);
    return null;
  }
};
