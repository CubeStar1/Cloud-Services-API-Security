const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

// Path to logs directory and log file
const logDir = path.join(__dirname, "logs");
const logFile = path.join(logDir, "traffic-logs.json");

// Path to Python executable (Ensure this path is correctly set in your environment)
const pythonExecutable = "python"; // Replace with actual path

// Path to Python script
const pythonScript = path.join(__dirname, "f1.py");

// Ensure the logs directory exists
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir);

// Function to log data to a file
function logToFile(content) {
  const logStream = fs.createWriteStream(logFile, { flags: "a" });
  logStream.write(JSON.stringify(content) + ",\n");
  logStream.end();
}

// Function to execute Python script
function handleGoogleDriveAccess() {
  console.log("Google Drive accessed - Running Python script...");

  exec(`${pythonExecutable} "${pythonScript}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error.message}`);
      return;
    }
    if (stderr) {
      console.error(`Python script stderr: ${stderr}`);
      return;
    }
    console.log(`Python script output:\n${stdout}`);
  });
}

// Export the module
module.exports = {
  *beforeSendRequest(requestDetail) {
    const host = requestDetail.requestOptions.headers["Host"] || "";

    const request = {
      type: "request",
      url: requestDetail.url,
      method: requestDetail.requestOptions?.method || "UNKNOWN",
      headers_Host: host,
      requestHeaders_Origin: requestDetail.requestOptions.headers["Origin"] || "",
      requestHeaders_Content_Type: requestDetail.requestOptions.headers["Content-Type"] || "",
      requestHeaders_Referer: requestDetail.requestOptions.headers["Referer"] || "",
      requestHeaders_Accept: requestDetail.requestOptions.headers["Accept"] || "",
      body: requestDetail.requestData ? requestDetail.requestData.toString() : null,
    };

    logToFile(request);

    if (host.includes("drive.google.com")) {
      handleGoogleDriveAccess();
    }

    return null;
  },

  *beforeSendResponse(requestDetail, responseDetail) {
    const host = requestDetail.url;

    const response = {
      type: "response",
      url: requestDetail.url,
      method: requestDetail.requestOptions?.method || "UNKNOWN",
      headers_Host: host,
      responseHeaders_Content_Type: responseDetail.response.header["Content-Type"] || "",
      body: responseDetail.response.body ? responseDetail.response.body.toString() : null,
    };

    logToFile(response);

    if (host.includes("cloud.google.com")) {
      handleGoogleDriveAccess();
    }

    return null;
  },
};