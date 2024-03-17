import axios from 'axios';
import { JSON_SERVER_URL } from '../utils/utils';

export const fetchUserContact = async (userId) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}`,
  };

  try {
    const response = await axios(requestOptions);
    return {
      status: response.status,
      email: response.data.email,
      phone: response.data.phone,
      firstName: response.data.firstName,
      lastName: response.data.lastName,
      profilePicture: response.data.profilePicture,
    };
  } catch (err) {
    return '';
  }
};

export const getBookmarks = async (
  userId,
  // limit = BOOKMARK_ITEM_PER_PAGE,
) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}/bookmarks`,
    // url: `${JSON_SERVER_URL}/bookmarks?userId=${userId}&_expand=post`,
  };

  try {
    const response = await axios(requestOptions);

    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};
