import { React, useState, useEffect } from "react";
import "./HomePage.css";
import Openai from '../components/Openai';

import GridImg from '../Images/grid.jpg';

import Sample1 from '../Images/Sample1.png';
import Sample2 from '../Images/Sample2.png';

import sendModelData from '../api/model.js';

function HomePage(props) {

  const [utility, setUtility] = useState('');
  const [region, setRegion] = useState('');
  const [size, setSize] = useState('');
  const [energy, setEnergy] = useState('');
  const [state, setState] = useState('');
  const [county, setCounty] = useState('');
  const [year, setYear] = useState(0);
  /// TODO: most likely remove year, internally pass it in with either 2024 or if it breaks, 2023
  //const [date, setDate] = useState(''); // YYYY-MM-DD

  const [show, setShow] = useState(0);
  //const [count, setCount] = useState(0);

  const [responseText, setResponseText] = useState('');


  const handleUtilityChange = (e) => {
    setUtility(e.target.value);
  };

  const handleRegionChange = (e) => {
    setRegion(e.target.value);
  };

  const handleSizeChange = (e) => {
    setSize(e.target.value);
  }

  const handleEnergyChange = (e) => {
    setEnergy(e.target.value);
  };

  const handleStateChange = (e) => {
    setState(e.target.value);
  };

  const handleCountyChange = (e) => {
    {/* change all lowercase*/ }
    setCounty(e.target.value.toLowerCase());
  };

  const handleYearChange = (e) => {
    setYear(e.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setUtility(document.getElementById('utilityTypeSelect').value);
    setRegion(document.getElementById('regionType').value);
    setEnergy(document.getElementById('energyType').value);
    setState(document.getElementById('stateSelect').value);
    //setEnergy(document.getElementById('energyType').value);
    //countyInput
    /// TODO: Logic to handle form submission, such as collecting form data

    try {
      const response = await sendModelData(utility, region, size, energy, state, county, year);
      const data = response.data
      {/*rec = data.recommendation;
      features = data.features;*/}
      console.log('Response from backend:', response);
      //setResponseText(data.analysis);
      //setResponseText(data.debug);
      setResponseText(data.txt);

      // Handle response as needed
    } catch (error) {
      console.error('Error:', error);
      // Handle error
    }
    setShow(1);



    //setCount(count + 1);

    /*
    if (count % 2 == 0) {
      setResponseText("Our model predicts that this project is unlikely to be withdrawn, with a 53% chance of no withdrawal. The three leading factors for this project, based on all of the parameters included, that have most influenced the prediction probability towards withdrawal are that the project is not located in the Southeast or Western regions of the United States, and that it is not a biomass project. Given the project parameters, it may be worthwhile to explore these options. According to our model, it is unlikely to face withdrawal because it is not one of the following types of energy: gravity rail, hybrid, hydroelectric, or offshore wind.");
    } else {
      setResponseText("Our model predicts that this project is likely to be withdrawn, with a 73% chance of withdrawal. The overwhelming factor in this decision is the region that it is built in (the Southeast of the United States), with an influencing value of 5.61 in our model, compared to 0.02 and 0.06 in the nearest factors. A project in the West of the United States may have fared better, as that was a leading feature that suggested that the project might not be withdrawn.");
    }
    */
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

      {/*
      <div className="row justify-content-center">
        <div className="col text-center">
          <Openai />
        </div>
      </div>
      */}

      {/* Form Section */}
      <div className="row justify-content-center">
        <div className="col-lg-8 col-md-10">
          <form onSubmit={handleSubmit} className="mt-5">
            <div className="form-group">
              <label htmlFor="utilityTypeSelect">Utility Type</label>
              <select id="utilityTypeSelect" name="utilityType" className="form-control" onChange={handleUtilityChange}>
                {utilityTypes.map((type, index) => (
                  <option key={index} value={type}>{capitalize(type)}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="regionType">Region Type</label>
              <select className="form-control" id="regionType" onChange={handleRegionChange}>
                <option>CAISO</option>
                <option>MISO</option>
                <option>PJM</option>
                <option>Southeast (non-ISO)</option>
                <option>West (non-ISO)</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="sizeInput">Project Size (in MW)</label>
              <input type="text" className="form-control" id="sizeInput" placeholder="Enter project size in MW" onChange={handleSizeChange} />
            </div>
            <div className="form-group">
              <label htmlFor="energyType">Energy Type</label>
              <select className="form-control" id="energyType" onChange={handleEnergyChange}>
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
                <option value="Landfill">Landfill</option>
                <option value="Methane">Methane</option>
                <option value="Nuclear">Nuclear</option>
                <option value="Offshore Wind">Offshore Wind</option>
                <option value="Oil">Oil</option>
                <option value="Pumped Storage">Pumped Storage</option>
                <option value="Solar">Solar</option>
                <option value="Steam">Steam</option>
                <option value="Waste Heat">Waste Heat</option>
                <option value="Wind">Wind</option>
                <option value="Wood">Wood</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="stateSelect">State</label>
              <select className="form-control" id="stateSelect" onChange={handleStateChange}>
                <option value="AL">Alabama</option>
                <option value="AK">Alaska</option>
                <option value="AZ">Arizona</option>
                <option value="AR">Arkansas</option>
                <option value="CA">California</option>
                <option value="CO">Colorado</option>
                <option value="CT">Connecticut</option>
                <option value="DE">Delaware</option>
                <option value="FL">Florida</option>
                <option value="GA">Georgia</option>
                <option value="HI">Hawaii</option>
                <option value="ID">Idaho</option>
                <option value="IL">Illinois</option>
                <option value="IN">Indiana</option>
                <option value="IA">Iowa</option>
                <option value="KS">Kansas</option>
                <option value="KY">Kentucky</option>
                <option value="LA">Louisiana</option>
                <option value="ME">Maine</option>
                <option value="MD">Maryland</option>
                <option value="MA">Massachusetts</option>
                <option value="MI">Michigan</option>
                <option value="MN">Minnesota</option>
                <option value="MS">Mississippi</option>
                <option value="MO">Missouri</option>
                <option value="MT">Montana</option>
                <option value="NE">Nebraska</option>
                <option value="NV">Nevada</option>
                <option value="NH">New Hampshire</option>
                <option value="NJ">New Jersey</option>
                <option value="NM">New Mexico</option>
                <option value="NY">New York</option>
                <option value="NC">North Carolina</option>
                <option value="ND">North Dakota</option>
                <option value="OH">Ohio</option>
                <option value="OK">Oklahoma</option>
                <option value="OR">Oregon</option>
                <option value="PA">Pennsylvania</option>
                <option value="RI">Rhode Island</option>
                <option value="SC">South Carolina</option>
                <option value="SD">South Dakota</option>
                <option value="TN">Tennessee</option>
                <option value="TX">Texas</option>
                <option value="UT">Utah</option>
                <option value="VT">Vermont</option>
                <option value="VA">Virginia</option>
                <option value="WA">Washington</option>
                <option value="WV">West Virginia</option>
                <option value="WI">Wisconsin</option>
                <option value="WY">Wyoming</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="countyInput">County Name</label>
              <input type="text" className="form-control" id="countyInput" placeholder="Enter your county name" onChange={handleCountyChange} />
            </div>
            <div className="form-group">
              <label htmlFor="year">Year</label>
              <input type="number" className="form-control" id="year" placeholder="Enter year" onChange={handleYearChange} />
            </div>
            <button type="submit" className="btn btn-primary">Submit</button>
          </form>
        </div>
      </div>

      <div>
        {show ? (
          <div className="container-fluid">
            <div className="row justify-content-center">
              <div className="col-12">
                <b>Response:</b>
              </div>
            </div>
            <div className="row justify-content-center">
              <div className="col-10">
                <p><b>Utility:</b> {utility}</p>
                <p><b>Region:</b> {region}</p>
                <p><b>Project Size:</b> {size} MW</p>
                <p><b>Energy:</b> {energy}</p>
                <p><b>State:</b> {state}</p>
                <p><b>County:</b> {county}</p>
                <p><b>Year:</b> {year}</p>

                {responseText}
              </div>
            </div>
          </div>) : ""
        }
      </div>

      {/*
      <div>
        {show ? (
          <div className="container-fluid">
            <div className="row justify-content-center">
              <div className="col-12">
                <img src={Sample1} alt="Image 0" className="img-fluid" />
              </div>
            </div>
          </div>) : ''
        }
      </div>
      */}

      {/* Sample Section */}
      <div class="top-bar"></div>
      <div id="samples-section" class="container">
        <h2 class="samples-heading">Samples</h2>
        <div class="row">
          <div class="col-md-6 sample-box">
            <h3>Sample 1</h3>
            <p>Our model predicts that this project is unlikely to be withdrawn, with a 53% chance of no withdrawal. The three leading factors for this project, based on all of the parameters included, that have most influenced the prediction probability towards withdrawal are that the project is not located in the Southeast or Western regions of the United States, and that it is not a biomass project. Given the project parameters, it may be worthwhile to explore these options. According to our model, it is unlikely to face withdrawal because it is not one of the following types of energy: gravity rail, hybrid, hydroelectric, or offshore wind.</p>
            <img src={Sample1} className="img-fluid mb-3" />
          </div>
          <div class="col-md-6 sample-box">
            <h3>Sample 2</h3>
            <p>Our model predicts that this project is likely to be withdrawn, with a 73% chance of withdrawal. The overwhelming factor in this decision is the region that it is built in (the Southeast of the United States), with an influencing value of 5.61 in our model, compared to 0.02 and 0.06 in the nearest factors. A project in the West of the United States may have fared better, as that was a leading feature that suggested that the project might not be withdrawn.</p>
            <img src={Sample2} className="img-fluid mb-3" />
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
