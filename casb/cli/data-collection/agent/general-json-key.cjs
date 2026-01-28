const fs = require('fs');
const path = require('path');

// Path to the logs directory and log file
const logDir = path.join(__dirname, 'logs');
const logFile = path.join(logDir, 'traffic-logs.json');

// Ensure the logs directory exists
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir);

// Helper function to log data to the file
function logToFile(content) {
  const logStream = fs.createWriteStream(logFile, { flags: 'a' });
  logStream.write(JSON.stringify(content) + ',\n');
  logStream.end();
}

// Helper function to check if content type should exclude body
function shouldExcludeBody(contentType) {
  if (!contentType) return false;
  const lowerContentType = contentType.toLowerCase();
  return (
    lowerContentType.includes('image/') ||
    lowerContentType.includes('application/json+protobuf') ||
    lowerContentType.includes('application/x-www-form-urlencoded')
  );
}

// Helper function to check if body content appears to be binary/encoded
function isBinaryOrEncoded(body) {
  if (!body) return false;
  // Check if the content has a high density of Unicode escape sequences or binary patterns
  const unicodePattern = /\\u[0-9a-f]{4}/i;
  const binaryIndicators = ['\u0000', '\u0001', '\u0002', '\u0003'];
  
  // If more than 10% of characters are Unicode escapes or contains binary indicators
  const unicodeMatches = (body.match(new RegExp(unicodePattern, 'g')) || []).length;
  const hasBinaryIndicators = binaryIndicators.some(indicator => body.includes(indicator));
  
  return hasBinaryMatches = unicodeMatches > body.length / 20 || hasBinaryIndicators;
}

module.exports = {

  // Intercept and log request details
  *beforeSendRequest(requestDetail) {
    const bodyContent = requestDetail.requestData ? requestDetail.requestData.toString() : null;
    const request = {
      type: 'request',
      url: requestDetail.url,
      method: requestDetail.requestOptions?.method || 'UNKNOWN',
      headers_Host: requestDetail.requestOptions.headers['Host'] || '',
      requestHeaders_Origin: requestDetail.requestOptions.headers['Origin'] || '',
      requestHeaders_Content_Type: requestDetail.requestOptions.headers['Content-Type'] || '',
      requestHeaders_Referer: requestDetail.requestOptions.headers['Referer'] || '',
      requestHeaders_Accept: requestDetail.requestOptions.headers['Accept'] || '',
      body: bodyContent && !isBinaryOrEncoded(bodyContent) ? bodyContent : null
    };

    logToFile(request);
    return null;
  },

  // Intercept and log response details
  *beforeSendResponse(requestDetail, responseDetail) {
    const contentType = responseDetail.response.header['Content-Type'] || '';
    const bodyContent = responseDetail.response.body ? responseDetail.response.body.toString() : null;
    const response = {
      type: 'response',
      url: requestDetail.url,
      method: requestDetail.requestOptions?.method || 'UNKNOWN',
      headers_Host: requestDetail.url,
      responseHeaders_Content_Type: contentType,
      body: (shouldExcludeBody(contentType) || (bodyContent && isBinaryOrEncoded(bodyContent))) ? null : bodyContent
    };

    logToFile(response);
    return null;
  }
};
