import { React, useState, useEffect } from "react";
//import { Link } from "react-router-dom";
//import { NAVBAR_HEIGHT } from "../utils/utils";
//import { displayRecent, displayUrgent } from "../api/home";
//import ItemPostingTable from "../components/ItemPostingTable";
//import { HOME_ITEM_PER_PAGE } from "../utils/utils";
//import PropTypes from "prop-types";

import "./Openai.css";

function Openai() {
  /*
  const inlineStyles = {
    height: NAVBAR_HEIGHT,
  };
  */

  /*
  const { userId } = props;
  const [recentResults, setRecentResults] = useState([]);
  const [urgentResults, setUrgentResults] = useState([]);
  const [pageRecent, setPageRecent] = useState(1);
  const [pageUrgent, setPageUrgent] = useState(1);
  const [numRecentResults, setNumRecentResults] = useState(0);
  const [numUrgentResults, setNumUrgentResults] = useState(0);
  */
  // const [inputText, setInputText] = useState('');
  // const [responseText, setResponseText] = useState('');

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

  // const handleInputChange = (e) => {
  //   setInputText(e.target.value);
  // };
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

  const [inputText, setInputText] = useState('');
  const [responseText, setResponseText] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  return (
    <div id="openai" className="container-fluid">
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
    </div>
  );
}

export default Openai;
