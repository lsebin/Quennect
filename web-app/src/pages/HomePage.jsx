import { React, useState, useEffect } from "react";
import "./HomePage.css";
import Openai from '../components/Openai';
import GridImg from '../Images/grid.jpg';

function HomePage(props) {
  return (
    <div id="home-page" className="container-fluid"> {/* Header */}
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
        <div className="col-md-4 text-center mb-3">
          <h3>Advanced Data Analytics</h3>
          <p>Transform complex datasets into strategic assets, optimize operations, and forecast trends with precision.</p>
        </div>
        <div className="col-md-4 text-center mb-3">
          <h3>Machine Learning</h3>
          <p>We use cutting edge machine learning algorithms to predict outcomes and automate decisions with increasing accuracy.</p>
        </div>
        <div className="col-md-4 text-center mb-3">
          <h3>Customized Solutions</h3>
          <p>We work with individual developers to understand their projects and the unique constraints involved to develop tailored solutions.</p>
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col text-center">
          <Openai />
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
