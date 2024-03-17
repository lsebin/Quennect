import axios from 'axios';
import { JSON_SERVER_URL } from '../utils/utils';

export const verifyLogin = async (username, password) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/login?username=${username}&password=${password}`,
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export const verifyToken = async (token, userId) => {
  const requestOptions = {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/login/verify-token`,
    data: {
      token,
      userId,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export const verifyId = async (username) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/login/new-user?username=${username}`,
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export const registerUser = async (
  username,
  password,
  firstName,
  lastName,
  email,
  question,
  answer,
) => {
  const requestOptions = {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/login/new-user`,

    data: {
      username,
      password,
      firstName,
      lastName,
      email,
      question,
      answer,
      header: {
        garagePhotoURL: '',
        garageDesc: '',
        moveoutDate: '',
        pickupLoc: '',
        avgRating: 0,
      },
    },
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export const verifyForgot = async (firstName, lastName, email) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/login/forgot?firstName=${firstName}&lastName=${lastName}&email=${email}`,
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export const resetPassword = async (password, user) => {
  const requestOptions = {
    method: 'put',
    headers: {
      'Content-Type': 'application/json',
    },
    // eslint-disable-next-line no-underscore-dangle
    url: `${JSON_SERVER_URL}/login/forgot/${user._id}`,

    // data: {
    //   ...user,
    //   password,
    //   id: undefined,
    // },

    data: {
      password,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message)
    return err.response;
  }
};
