import React, { useEffect, useState } from "react";
import axios from "axios";
//import logo from "./logo.svg";
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from './components/Navbar';
import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import { NAVBAR_HEIGHT } from './utils/utils';

function App() {
  //const navigate = useNavigate();
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div style={{ height: NAVBAR_HEIGHT }} />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<HomePage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
