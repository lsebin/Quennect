import { React, useState, useEffect } from "react";

import Alina from '../Images/Alina.png';
import Soyoon from '../Images/Soyoon.png';
import Caroline from '../Images/Caroline.png'
import Shane from '../Images/Shane.png'
import Sebin from '../Images/Sebin.png'

//import PropTypes from "prop-types";
//import { displayRecent, displayUrgent } from "../api/home";
import "./AboutPage.css";
//import './MyProfilePage.css';
//import ItemPostingTable from "../components/ItemPostingTable";
import { HOME_ITEM_PER_PAGE } from "../utils/utils";


function AboutPage(props) {
  /*
  const { userId } = props;
  const [recentResults, setRecentResults] = useState([]);
  const [urgentResults, setUrgentResults] = useState([]);
  const [pageRecent, setPageRecent] = useState(1);
  const [pageUrgent, setPageUrgent] = useState(1
  const [numRecentResults, setNumRecentResults] = useState(0);
  const [numUrgentResults, setNumUrgentResults] = useState(0);

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
    <div id="about-page" className="container-fluid">
      <div className="row justify-content-center text-center mb-5">
        <div className="col-8">
          <h2>About Quennect</h2>
          <p><b>Quennect</b> is designed to assist developers in navigating
            the interconnection queue by predicting the likelihood of a
            project’s progression, estimating wait times, and suggesting
            parameter changes to expedite the process. The interconnection
            queue system is a critical yet inefficient process for integrating
            new energy generation and storage projects into the transmission
            grid. The project’s goal is to enhance the efficiency of the
            development and operation stages of renewable energy projects by
            providing data-driven insights and recommendations.</p>
        </div>
      </div>
      <div className="row justify-content-center">
        {/* Updated profile cards for each member */}
        <div className="col-3 profile-card">
          <img src={Alina} alt="Alina Ho" className="img-fluid img-profile" />
          <div className="profile-name">ALINA HO</div>
          <div className="profile-role">Economics and Systems Engineering</div>
          <div className="profile-email">alinaho@seas.upenn.edu</div>
        </div>
        <div className="col-3 profile-card">
          <img src={Soyoon} alt="Soyoon Park" className="img-fluid img-profile" />
          <div className="profile-name">SOYOON PARK</div>
          <div className="profile-role">Computer Engineering and Mathematics</div>
          <div className="profile-email">soyoon@seas.upenn.edu</div>
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-3 profile-card">
          <img src={Shane} alt="Shane Williams" className="img-fluid img-profile" />
          <div className="profile-name">SHANE WILLIAMS</div>
          <div className="profile-role">Systems Engineering</div>
          <div className="profile-email">shanetw@seas.upenn.edu</div>
        </div>
        <div className="col-3 profile-card">
          <img src={Caroline} alt="Caroline Magdolen" className="img-fluid img-profile" />
          <div className="profile-name">CAROLINE MAGDOLEN</div>
          <div className="profile-role">Systems Engineering and Environmental Science</div>
          <div className="profile-email">magdolen@seas.upenn.edu</div>
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-3 profile-card">
          <img src={Sebin} alt="Sebin Lee" className="img-fluid img-profile" />
          <div className="profile-name">SEBIN LEE</div>
          <div className="profile-role">Computer Science</div>
          <div className="profile-email">seblee@seas.upenn.edu</div>
        </div>
      </div>
    </div>
  );
}

export default AboutPage;
/*
HomePage.propTypes = {
  userId: PropTypes.string.isRequired,
};
*/