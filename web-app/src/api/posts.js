import axios from 'axios';
import { JSON_SERVER_URL } from '../utils/utils';

export const createNewPost = async (postObj) => {
  const requestOptions = {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/posts`,
    data: {
      sellerId: postObj.sellerId,
      title: postObj.title,
      seller: postObj.seller,
      description: postObj.description,
      image: postObj.image,
      priceType: postObj.priceType,
      price: postObj.price,
      postedOn: postObj.postedOn,
      availableUntil: postObj.availableUntil,
      category: postObj.category,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response.data.insertedId;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};

export const fetchAllUsersPosts = async (ownerId) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/posts?sellerId=${ownerId}`,
  };

  try {
    // console.log('fetchAllUsersPosts requestOptions', requestOptions);
    const response = await axios(requestOptions);
    // console.log('fetchAllUsersPosts response', response.data.reverse());
    // The reverse call will put the most recent posts first.
    return response.data;
  } catch (err) {
    // console.error(err.message);
    return err.data.data;
  }
};

export const updateBid = async (postId, price, highestBidderId) => {
  const requestOptions = {
    method: 'patch',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/posts/bid/${postId}`,
    data: {
      price,
      highestBidderId,
    },
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    return err;
  }
};

export const editPost = async (postId, postObjUpdate) => {
  const requestOptions = {
    method: 'patch',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/posts/${postId}`,
    data: {
      title: postObjUpdate.title,
      description: postObjUpdate.description,
      image: postObjUpdate.image,
      priceType: postObjUpdate.priceType,
      price: postObjUpdate.price,
      availableUntil: postObjUpdate.availableUntil,
      category: postObjUpdate.category,
    },
  };

  try {
    // console.log('editPost requestOptions', requestOptions);
    const response = await axios(requestOptions);
    // console.log('editPost response', response);
    return response.status;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};

export const deletePost = async (postId) => {
  const requestOptions = {
    method: 'delete',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/posts/${postId}`,
  };

  try {
    // console.log('deletePost requestOptions', requestOptions);
    const response = await axios(requestOptions);
    // console.log('deletePost response', response);
    return response.status;
  } catch (err) {
    // console.error(err.message);
    return err;
  }
};
