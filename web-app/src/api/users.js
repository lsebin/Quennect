import axios from 'axios';
import { JSON_SERVER_URL } from '../utils/utils';

export const fetchUserInfo = async (userId) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}`,
  };

  try {
    const response = await axios(requestOptions);
    const { firstName: fName, lastName: lName, header: head } = response.data;

    const output = {
      firstName: fName,
      lastName: lName,
      header: head,
    };
    return output;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};

export const editHeaderInfo = async (userId, headerObj) => {
  const requestOptions = {
    method: 'patch',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}`,
    data: {
      header: headerObj,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response.status;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};

export const editUserInfo = async (
  userId,
  pw,
  firstname,
  lastname,
  email,
  phone,
  question,
  answer,
  url,
) => {
  const requestOptions = {
    method: 'patch',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}`,
    data: {
      firstName: firstname,
      lastName: lastname,
      email,
      password: pw,
      phone,
      question,
      answer,
      profilePicture: url,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};

export const currentUserInfo = async (userId) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}`,
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};
