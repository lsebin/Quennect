import { React, useState, useEffect } from "react";
import "./HomePage.css";
import Openai from '../components/Openai';

import GridImg from '../Images/grid.jpg';
import Sample1 from '../Images/Sample1.png';
import Sample2 from '../Images/Sample2.png';

function HomePage(props) {
  const handleSubmit = (event) => {
    event.preventDefault();
    // Logic to handle form submission, such as collecting form data
  };

  const utilityTypes = [
    'aec', 'aep', 'ameren illinois', 'ameren missouri', 
    'ameren transmission company of illinois', 'american transmission co. llc', 
    'aps', 'atsi', 'avista', 'bge', 'bhct', 'big rivers electric corporation', 
    'cedar falls utilities', 'clpt', 'comed', 'commonwealth edison company', 
    'consumers energy company', 'cooperative energy', 'dairyland power cooperative', 
    'dayton', 'dcrt', 'deok', 'dl', 'dominion', 'dominion sc', 'dominionsc', 'dpl', 
    'duke', 'duke energy', 'duke energy corporation', 'duke energy indiana, llc', 
    'east texas electric cooperative', 'entergy', 'entergy arkansas, llc', 
    'entergy louisiana, llc', 'entergy mississippi, llc', 'entergy texas, inc.', 
    'ewington energy systems, llc', 'firstenergy solutions corp.', 'glw', 
    'great river energy', 'gridunity', 'gtc', 'hoosier energy rec inc', 
    'hoosier energy rec, inc.', 'indianapolis power & light company', 'itc midwest', 
    'itc transmission', 'itci', 'jcpl', 'madison gas and electric company', 'me', 
    'michigan electric transmission company, llc', 'michigan public power agency', 
    'midamerican energy company', 'minnesota power', 'minnesota power (allete, inc.)', 
    'missouri river energy services (mres)', 'montana-dakota utilities co.', 
    'montana-dakota utilities company', 'northern indiana public service company', 
    'northern states power (xcel energy)', 'northwestern', 'nve', 
    'otter tail power company', 'ovec', 'pacificorp', 'peco', 'penelec', 'pepco', 
    'pgae', 'pge', 'ppl', 'prairie power, inc.', 'pse', 'pseg', 'reco', 
    'rochester public utilities', 'sce', 'sdge', 'smeco', 'smmpa', 'soco', 
    'south mississippi electric power association', 
    'southern illinois power cooperative', 
    'southern indiana gas & electric company d/b/a vectren energy delivery of indiana, inc.', 
    'southern minnesota municipal power agency', 'srp', 'srp-anpp', 'srp-pv-pc', 
    'tsgt', 'ugi', 'unknown', 'upper peninsula power company', 'vea', 
    'wabash valley power', 'wisconsin electric power company', 
    'wisconsin public service corporation', 'wolverine power supply cooperative, inc.', 
    'xcel energy'
  ];

  // Function to capitalize the first letter of each word for display
  const capitalize = (s) => s.replace(/\b\w/g, letter => letter.toUpperCase());


  return (
    <div id="home-page" className="container-fluid"> {/* Header */}
    <div class="top-bar"></div>
      <div className="row justify-content-center mb-4">
        <div className="col-md-8 text-center">
          <h2 className="mb-3">A Commitment to Innovation and Sustainability</h2>
          <p>Quennect aims to remove the speculation from the development process by arming renewable project developers with the ability to set themselves up for success.</p>
          <img src={GridImg} alt="A commitment to innovation and sustainability" className="img-fluid mb-3" />
        </div>
      </div>
      <div className="row justify-content-center mb-4"> {/* Approach */}
        <div className="col-md-8 text-center">
          <h2 className="mb-3">Data-Driven Decision Making</h2>
          <p>Quennect is designed to assist developers in navigating the interconnection queue by predicting the likelihood of a project’s progression, estimating wait times, and suggesting parameter changes to expedite the process.</p>
        </div>
      </div>
      <div className="row justify-content-center mb-4"> {/* Methodology */}
        <div class="col-md-4 text-center mb-3 subsection-box">
          <h3>Advanced Data Analytics</h3>
          <p>Transform complex datasets into strategic assets, optimize operations, and forecast trends with precision.</p>
        </div>
        <div class="col-md-4 text-center mb-3 subsection-box">
          <h3>Machine Learning</h3>
          <p>We use cutting edge machine learning algorithms to predict outcomes and automate decisions with increasing accuracy.</p>
        </div>
        <div class="col-md-4 text-center mb-3 subsection-box">
          <h3>Customized Solutions</h3>
          <p>We work with individual developers to understand their projects and the unique constraints involved to develop tailored solutions.</p>
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col text-center">
          <Openai />
        </div>
      </div>
      
    {/* Form Section */}
    <div className="row justify-content-center">
        <div className="col-lg-8 col-md-10">
          <form onSubmit={handleSubmit} className="mt-5">
            <div className="form-group">
              <label htmlFor="utilityTypeSelect">Utility Type</label>
              <select id="utilityTypeSelect" name="utilityType" className="form-control">
                {utilityTypes.map((type, index) => (
                  <option key={index} value={type}>{capitalize(type)}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="regionType">Region Type</label>
              <select className="form-control" id="regionType">
                <option>CAISO</option>
                <option>MISO</option>
                <option>PJM</option>
                <option>Southeast (non-ISO)</option>
                <option>West (non-ISO)</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="energyType">Energy Type</label>
              <select className="form-control" id="energyType">
                <option value="Battery">Battery</option>
                <option value="Biofuel">Biofuel</option>
                <option value="Biogas">Biogas</option>
                <option value="Biomass">Biomass</option>
                <option value="Coal">Coal</option>
                <option value="Diesel">Diesel</option>
                <option value="Flywheel">Flywheel</option>
                <option value="Gas">Gas</option>
                <option value="Geothermal">Geothermal</option>
                <option value="Gravity Rail">Gravity Rail</option>
                <option value="Hybrid">Hybrid</option>
                <option value="Hydro">Hydro</option>
                <option value="Methane">Methane</option>
                <option value="Nuclear">Nuclear</option>
                <option value="Offshore Wind">Offshore Wind</option>
                <option value="Oil">Oil</option>
                <option value="Pumped Storage">Pumped Storage</option>
                <option value="Solar">Solar</option>
                <option value="Waste Heat">Waste Heat</option>
                <option value="Wind">Wind</option>
                <option value="Wood">Wood</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="latitude">Latitude</label>
              <input type="number" className="form-control" id="latitude" placeholder="Enter latitude" />
            </div>
            <div className="form-group">
              <label htmlFor="longitude">Longitude</label>
              <input type="number" className="form-control" id="longitude" placeholder="Enter longitude" />
            </div>
            <div className="form-group">
              <label htmlFor="year">Year</label>
              <input type="number" className="form-control" id="year" placeholder="Enter year" />
            </div>
            <div className="form-group">
              <label htmlFor="date">Date</label>
              <input type="date" className="form-control" id="date" />
            </div>
            <button type="submit" className="btn btn-primary">Submit</button>
          </form>
        </div>
      </div>
      
    {/* Sample Section */}
    <div class="top-bar"></div>
    <div id="samples-section" class="container">
    <h2 class="samples-heading">Samples</h2>
      <div class="row">
        <div class="col-md-6 sample-box">
          <h3>Sample 1</h3>
          <p>Our model predicts that this project is unlikely to be withdrawn, with a 53% chance of no withdrawal. The three leading factors for this project, based on all of the parameters included, that have most influenced the prediction probability towards withdrawal are that the project is not located in the Southeast or Western regions of the United States, and that it is not a biomass project. Given the project parameters, it may be worthwhile to explore these options. According to our model, it is unlikely to face withdrawal because it is not one of the following types of energy: gravity rail, hybrid, hydroelectric, or offshore wind.</p>
          <img src={Sample1}className="img-fluid mb-3" />
        </div>
        <div class="col-md-6 sample-box">
          <h3>Sample 2</h3>
          <p>Our model predicts that this project is likely to be withdrawn, with a 73% chance of withdrawal. The overwhelming factor in this decision is the region that it is built in (the Southeast of the United States), with an influencing value of 5.61 in our model, compared to 0.02 and 0.06 in the nearest factors. A project in the West of the United States may have fared better, as that was a leading feature that suggested that the project might not be withdrawn.</p>
          <img src={Sample2}className="img-fluid mb-3" />
        </div>
      </div>
    </div>
    </div>
  );
}

export default HomePage;

/*
HomePage.propTypes = {
  userId: PropTypes.string.isRequired,
};
*/
