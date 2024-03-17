import axios from 'axios';
import { JSON_SERVER_URL } from '../utils/utils';

export const createNewMessage = async (userId, receiverId, userMessage) => {
  const requestOptions = {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/messages`,
    data: {
      userId,
      receiverId,
      userMessage,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    return err;
  }
};

export const getUserReceiverMessages = async (userId, receiverId) => {
  const requestOptions1 = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/messages?userId=${userId}&receiverId=${receiverId}`,
  };

  try {
    const response1 = await axios(requestOptions1);
    return response1;
  } catch (err) {
    return err;
  }
};

export const getChatRecipients = async (userId) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/messages/recipients?userId=${userId}`,
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    return err;
  }
};
