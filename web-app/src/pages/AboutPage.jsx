import { React, useState, useEffect } from "react";
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
  const [pageUrgent, setPageUrgent] = useState(1);
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
      <div className="row justify-content-center">
        <div className="col-6">
          <b>About</b>
          <br />
          <b>Quennect</b> is designed to assist developers in navigating
          the interconnection queue by predicting the likelihood of a
          project’s progression, estimating wait times, and suggesting
          parameter changes to expedite the process. The interconnection
          queue system is a critical yet inefficient process for integrating
          new energy generation and storage projects into the transmission
          grid. The project’s goal is to enhance the efficiency of the
          development and operation stages of renewable energy projects by
          providing data-driven insights and recommendations.
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-3">
          <b>ALINA HO</b>
          <br />
          Economics and Systems Engineering
          <br />
          <i>School of Engineering and Applied Science</i>
          <br />
          alinaho@seas.upenn.edu
        </div>
        <div className="col-3">
          <b>SOYOON PARK</b>
          <br />
          Computer Engineering and Mathematics
          <br />
          <i>School of Engineering and Applied Science</i>
          <br />
          soyoon@seas.upenn.edu
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-3">
          <b>SHANE WILLIAMS</b>
          <br />
          Systems Engineering
          <br />
          <i>School of Engineering and Applied Science</i>
          <br />
          shanetw@seas.upenn.edu
        </div>
        <div className="col-3">
          <b>CAROLINE MAGDOLEN</b>
          <br />
          Systems Engineering and Environmental Science
          <br />
          <i>School of Engineering and Applied Science</i>
          <br />
          magdolen@seas.upenn.edu
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-3">
          <b>SEBIN LEE</b>
          <br />
          Computer Science
          <br />
          <i>School of Engineering and Applied Science</i>
          <br />
          seblee@seas.upenn.edu
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