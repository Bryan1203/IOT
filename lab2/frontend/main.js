const { app, BrowserWindow } = require('electron')

// include the Node.js 'path' module at the top of your file
const path = require('node:path')

// modify your existing createWindow() function
function createWindow () {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
      width: 1000,
      height: 1000,
      webPreferences: {
        nodeIntegration: true, //you might need to add this too
        contextIsolation:false, // Add this parameter setting
        preload: path.join(__dirname, 'preload.js')
      }
    })
  
    mainWindow.loadFile('index.html')
}

  app.whenReady().then(() => {
    createWindow()
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
  })