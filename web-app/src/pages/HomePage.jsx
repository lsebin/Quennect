import { React, useState, useEffect } from "react";
import "./HomePage.css";
import Openai from '../components/Openai';

function HomePage(props) {
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
      <Openai />
    </div >
  );
}

export default HomePage;

/*
HomePage.propTypes = {
  userId: PropTypes.string.isRequired,
};
*/
