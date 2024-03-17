import React from "react";
import { Link } from "react-router-dom";
import { NAVBAR_HEIGHT } from "../utils/utils";
import "./Navbar.css";

function Navbar() {
  const inlineStyles = {
    height: NAVBAR_HEIGHT,
  };

  return (
    <div id="navbar" style={inlineStyles} className="container-fluid">
      <div className="row">
        <Link to="/" id="logo" className="col-lg-2 col-3">
          {" "}
          Quennect{" "}
        </Link>
        <nav id="nav-links" className="col-lg-10 col-9">
          <Link to="/about"> About </Link>
        </nav>
      </div>
    </div>
  );
}

export default Navbar;
