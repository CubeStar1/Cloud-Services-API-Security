const fs = require('fs');
const path = require('path');

// Base path and file configuration
const BASE_PATH = path.resolve(__dirname, '../');  
const LOG_FILE_NAME = process.env.LOG_FILE_NAME || 'box-traffic-logs.json';
const LOG_DIR = path.join(BASE_PATH, 'data', 'logs', 'raw-json');
const LOG_FILE = path.join(LOG_DIR, LOG_FILE_NAME);

// Debug: Print paths to verify
console.log('Using log file:', LOG_FILE);
console.log('Base Path:', BASE_PATH);
console.log('Log Directory:', LOG_DIR);
console.log('Log File:', LOG_FILE);

// MIME types to skip body content
const SKIP_BODY_MIME_TYPES = [
    'image/',           
    'application/octet-stream',
    'text/html',
    'text/css',
    'application/javascript',
    'application/x-javascript',
    'text/javascript',
    'application/x-protobuf'
];

// Ensure the logs directory exists
try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    console.log('Successfully created/verified log directory:', LOG_DIR);
} catch (error) {
    console.error('Error creating log directory:', error);
}

// Helper function to check if body should be skipped based on MIME type
function shouldSkipBody(contentType) {
    if (!contentType) return false;
    return SKIP_BODY_MIME_TYPES.some(mimeType => contentType.toLowerCase().startsWith(mimeType));
}

// Helper function to check if content appears to be binary or buffer data
function isBinaryOrBufferContent(body) {
    if (!body) return false;
    
    // Check if it's a Buffer instance
    if (Buffer.isBuffer(body)) return true;
    
    // If it's a string, try to parse it
    if (typeof body === 'string') {
        try {
            const parsed = JSON.parse(body);
            // Check for common binary/buffer indicators in the parsed content
            if (parsed && typeof parsed === 'object') {
                // Check for Buffer-like objects
                if (parsed.type === 'Buffer' || parsed.data instanceof Uint8Array) return true;
                // Check for objects with 'type' indicating binary content
                if (parsed.type === 'Binary' || parsed.type === 'Buffer') return true;
                // Check for base64 encoded data
                if (parsed.data && typeof parsed.data === 'string' && 
                    /^[A-Za-z0-9+/=]{20,}$/.test(parsed.data)) return true;
            }
        } catch (e) {
            // If it's not JSON, check if it looks like binary data
            return /[\x00-\x08\x0B\x0C\x0E-\x1F]/.test(body);
        }
    }
    
    return false;
}

// Helper function to safely parse and sanitize body content
function sanitizeBody(body, contentType) {
    if (!body) return null;
    
    // Skip if content appears to be binary or buffer data
    if (isBinaryOrBufferContent(body)) {
        console.log('Skipping binary/buffer content');
        return null;
    }
    
    try {
        // If it's JSON content, try to parse and re-stringify it properly
        if (contentType.includes('application/json')) {
            // Handle the case where body might be already parsed
            const parsed = typeof body === 'string' ? JSON.parse(body) : body;
            // Skip if the parsed content contains binary data
            if (isBinaryOrBufferContent(parsed)) return null;
            return JSON.stringify(parsed);
        }
        
        // For text content, clean up escape sequences and normalize
        if (contentType.startsWith('text/')) {
            const cleanText = body.toString()
                .replace(/\\n/g, ' ')           // Replace newlines with spaces
                .replace(/\\t/g, ' ')           // Replace tabs with spaces
                .replace(/\s+/g, ' ')           // Normalize whitespace
                .replace(/\\"/g, '"')           // Fix escaped quotes
                .trim();
            return cleanText;
        }
        
        // For other content types, just convert to string if possible
        return typeof body === 'object' ? JSON.stringify(body) : body.toString();
    } catch (error) {
        console.error('Error sanitizing body:', error);
        return null;  // Return null if we can't properly sanitize the content
    }
}

// Helper function to log data to the file and console
function logToFile(content) {
    try {
        // Ensure the content is properly stringified
        const logData = typeof content === 'string' ? content : JSON.stringify(content);
        const logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' });
        logStream.write(logData + ',\n');
        logStream.end();
        
        // Output structured log to console
        const structuredLog = {
            timestamp: new Date().toISOString(),
            type: 'log',
            data: content
        };
        console.log(JSON.stringify(structuredLog));
    } catch (error) {
        console.error('Error writing to log file:', error);
    }
}

module.exports = {
    // Intercept and log request details
    *beforeSendRequest(requestDetail) {
        const contentType = requestDetail.requestOptions.headers['Content-Type'] || '';
        
        const request = {
            type: 'request',
            url: requestDetail.url,
            method: requestDetail.requestOptions?.method || 'UNKNOWN',
            headers_Host: requestDetail.requestOptions.headers['Host'] || '',
            requestHeaders_Origin: requestDetail.requestOptions.headers['Origin'] || '',
            requestHeaders_Content_Type: contentType,
            requestHeaders_Referer: requestDetail.requestOptions.headers['Referer'] || '',
            requestHeaders_Accept: requestDetail.requestOptions.headers['Accept'] || '',
            // Sanitize body content if not skipped
            body: shouldSkipBody(contentType) ? null :
                  sanitizeBody(requestDetail.requestData, contentType)
        };

        logToFile(request);
        return null;
    },

    // Intercept and log response details
    *beforeSendResponse(requestDetail, responseDetail) {
        const contentType = responseDetail.response.header['Content-Type'] || '';
        
        const response = {
            type: 'response',
            url: requestDetail.url,
            method: requestDetail.requestOptions?.method || 'UNKNOWN',
            headers_Host: requestDetail.url,
            responseHeaders_Content_Type: contentType,
            // Sanitize body content if not skipped
            body: shouldSkipBody(contentType) ? null : 
                  sanitizeBody(responseDetail.response.body, contentType)
        };

        logToFile(response);
        return null;
    }
};
