import { React, useState, useEffect } from "react";
import submit from "../api/openai";
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

  /*
  const [formData, setFormData] = useState({
    region: '',
    email: '',
    message: ''
  });
  */

  /*
  //const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };
  */


  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const callSubmit = async (e) => {
    e.preventDefault();

    const response = await submit(
      inputText
    );

    if (response.data) {
      setResponseText(response.data.choices[0].message.content);
    }

    if (response.status === 201) {
      // setErrMsg('Registration complete!');
      // setShow(true);
    } else {
      // setErrMsg('Error occurred.');
    }
    // console.log(response.data);
  };

  /*
  //<form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label for="email" clsas="form-label"> Email address </label>
          <input
            type="email"
            className="form-control"
            id="email"
            value={formData.email}
            onChange={ }
            placeholder="name@example.com" />
        </div>
      </form>
       <textarea
          value={inputText}
          //onChange={handleInputChange}
          placeholder="Enter your input here..."
          rows={4}
          cols={50}
        />
        <br />
      <div class="dropdown">
        <button class="btn btn-secondary dropdown-toggle" type="button" id="dropdownMenuButton" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
          Region
        </button>
        <div class="dropdown-menu" aria-labelledby="dropdownMenuButton">
          <a class="dropdown-item" href="#">Action</a>
          <a class="dropdown-item" href="#">Another action</a>
          <a class="dropdown-item" href="#">Something else here</a>
        </div>
      </div>
  */

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
      <div className="row">
        <div className="col-12">
          <button type="submit" className="btn btn-secondary" onClick={callSubmit}>
            Submit
          </button>
        </div>
      </div>
      {responseText && (
        <div className="container-fluid">
          <div className="row justify-content-center">
            <div className="col-12">
              <b>Response:</b>
            </div>

          </div>
          <div className="row justify-content-center">
            <div className="col-10">
              {responseText}
            </div>
          </div>
        </div>)
      }
    </div>


  );
}

export default Openai;
