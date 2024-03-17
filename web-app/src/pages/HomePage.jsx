import { React, useState, useEffect } from "react";
import PropTypes from "prop-types";
//import { displayRecent, displayUrgent } from "../api/home";
import "./HomePage.css";
//import './MyProfilePage.css';
//import ItemPostingTable from "../components/ItemPostingTable";
//import { HOME_ITEM_PER_PAGE } from "../utils/utils";

//function HomePage(props) {
function HomePage(props) {
  /*
  const { userId } = props;
  const [recentResults, setRecentResults] = useState([]);
  const [urgentResults, setUrgentResults] = useState([]);
  const [pageRecent, setPageRecent] = useState(1);
  const [pageUrgent, setPageUrgent] = useState(1);
  const [numRecentResults, setNumRecentResults] = useState(0);
  const [numUrgentResults, setNumUrgentResults] = useState(0);
  */
  const [inputText, setInputText] = useState('');
  const [responseText, setResponseText] = useState('');

  /*
  useEffect(() => {
    const callDisplayRecent = async () => {
      const responseRecent = await displayRecent(pageRecent);
      if (responseRecent.status === 200 || responseRecent.status === 404) {
        setPageRecent(1);
        setNumRecentResults(
          parseInt(responseRecent.headers["x-total-count"], 10)
        );
        setRecentResults(responseRecent.data);
      }
    };

    const callDisplayUrgent = async () => {
      const responseUrgent = await displayUrgent(pageUrgent);
      if (responseUrgent.status === 200 || responseUrgent.status === 404) {
        setPageUrgent(1);
        setNumUrgentResults(
          parseInt(responseUrgent.headers["x-total-count"], 10)
        );
        setUrgentResults(responseUrgent.data);
      }
    };

    callDisplayRecent();
    callDisplayUrgent();
  }, []);
  */

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };
  /*
  const handleRecentNextPage = async () => {
    const newPage = pageRecent + 1;
    const response = await displayRecent(newPage);
    if (response.status === 200 || response.status === 404) {
      setPageRecent(newPage);
      setNumRecentResults(parseInt(response.headers["x-total-count"], 10));
      setRecentResults(response.data);
    }
  };
 
  const handleRecentPrevPage = async () => {
    const newPage = pageRecent - 1;
    const response = await displayRecent(newPage);
 
    if (response.status === 200 || response.status === 404) {
      setPageRecent(newPage);
      setNumRecentResults(parseInt(response.headers["x-total-count"], 10));
      setRecentResults(response.data);
    }
  };
 
  const handleUrgentNextPage = async () => {
    const newPage = pageUrgent + 1;
    const response = await displayUrgent(newPage);
 
    if (response.status === 200 || response.status === 404) {
      setPageUrgent(newPage);
      setNumUrgentResults(parseInt(response.headers["x-total-count"], 10));
      setUrgentResults(response.data);
    }
  };
  const handleUrgentPrevPage = async () => {
    const newPage = pageUrgent - 1;
    const response = await displayUrgent(newPage);
 
    if (response.status === 200 || response.status === 404) {
      setPageUrgent(newPage);
      setNumUrgentResults(parseInt(response.headers["x-total-count"], 10));
      setUrgentResults(response.data);
    }
    
};
*/

  return (
    <div id="home-page" className="container-fluid">
      <div className="row justify-content-center">
        <div className="col-6">
          <b>A commitment to innovation and sustainability</b>
          <br />
          Quennect aims to remove the speculation from the development process by arming renewable project developers with the ability to set themselves up for success
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-6">
          <b>Data-Driven Decision Making</b>
          <br />
          Quennect is designed to assist developers in navigating the interconnection queue by predicting the likelihood of a project’s progression, estimating wait times, and suggesting parameter changes to expedite the process
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-2">
          <b>Advanced Data Analytics
          </b>
          <br />
          Transform complex datasets into strategic assets, optimize operations, and forecast trends with precision.
        </div>
        <div className="col-2">
          <b>Machine Learning</b>
          <br />
          We use cutting edge machine learning algrothms to predict outcomes and automate decisions with increasing accuracy.
        </div>
        <div className="col-2">
          <b>Customized Solutions</b>
          <br />
          We work with indivdual developers to understand their projects and the unqiue constraints invovled to develop tailored solutions.
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-3">
          Input prompt:
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-6">
          <input type="text" name="input-query" placeholder="enter input text here" value={inputText} onChange={handleInputChange} />
        </div>
      </div>
    </div >
  );
}

export default HomePage;

/*
HomePage.propTypes = {
  userId: PropTypes.string.isRequired,
};
*/
