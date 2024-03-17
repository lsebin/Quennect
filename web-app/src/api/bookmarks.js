import axios from 'axios';
import { JSON_SERVER_URL } from '../utils/utils';

const createNewBookmark = async (userId, postId) => {
  const requestOptions = {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/users/${userId}/bookmarks`,
    data: {
      userId,
      postId,
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

export default createNewBookmark;
