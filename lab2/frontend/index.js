var server_port = 65432;
var server_addr = "192.168.68.81";   // the IP address of your Raspberry PI

function client(){
    
    const net = require('net');
    var input = document.getElementById("myName").value;

    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        // send the message
        client.write(`${input}\r\n`);
    });
    
    // get the data from the server
    client.on('data', (data) => {
        document.getElementById("greet_from_server").innerHTML = data;
        console.log(data.toString());
        client.end();
        client.destroy();
    });

    client.on('end', () => {
        console.log('disconnected from server');
    });


}

// function toServer(input,id){
//     const net = require('net');
//     const client = net.createConnection({ port: server_port, host: server_addr }, () => {
//         // 'connect' listener.
//         console.log('connected to server!');
//         // send the message
//         client.write(`${input}\r\n`);
//     });
    
//     // get the data from the server
//     client.on('data', (data) => {
//         document.getElementById(id).innerHTML = data;
//         console.log(data.toString());
//         client.end();
//         client.destroy();
//     });

//     client.on('end', () => {
//         console.log('disconnected from server');
//     });
// }


function greeting(){

    // get the element from html
    var name = document.getElementById("myName").value;
    // update the content in html
    document.getElementById("greet").innerHTML = "Hello " + name + " !";
    // send the data to the server 
    //to_server(name);
    client();
    

}

function updateDisplay(speed, orientation, distance) {
    document.getElementById('speed').textContent = speed;
    document.getElementById('orientation').textContent = orientation;
    document.getElementById('distance').textContent = distance;
  }


  // Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBAHEZcJ8F1WiusYmaw_8rZD2exXFwVW2I",
  authDomain: "hydrobryan-608df.firebaseapp.com",
  databaseURL: "https://hydrobryan-608df-default-rtdb.firebaseio.com",
  projectId: "hydrobryan-608df",
  storageBucket: "hydrobryan-608df.appspot.com",
  messagingSenderId: "112798735952",
  appId: "1:112798735952:web:716001891a223a9f08bb55",
  measurementId: "G-0BLJR0C008"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);




function toServer(input,id){
    // document.getElementById(id).innerHTML = "hello";
    // const net = require('net');
    // const client = net.createConnection({ port: server_port, host: server_addr }, () => {
    //     // 'connect' listener.
    //     console.log('connected to server!');
    //     // send the message
    //     client.write(`${input}\r\n`);
    // });
    
    // // get the data from the server
    // client.on('data', (data) => {
    //     document.getElementById(id).innerHTML = data;
    //     console.log(data.toString());
    //     client.end();
    //     client.destroy();
    // });

    // client.on('end', () => {
    //     console.log('disconnected from server');
    // });

    console.log("helko");

    database.ref('UsersData/8D1wS1ZFjUOIOQg6m531banMEgE3/readings').once('litPerMin')
  .then((snapshot) => {
    // Handle the retrieved data
    const readings = snapshot.val();
    console.log(readings);
    
    // Access specific reading data
    const timestamp = '1711929186';
    const reading = readings[timestamp][0];
    console.log('Battery Voltage:', reading.batteryVolt);
    console.log('Liters per Minute:', reading.litPerMin);
    console.log('Timestamp:', reading.timestamp);
  })
  .catch((error) => {
    // Handle any errors
    console.error('Error retrieving data:', error);
  });
}
