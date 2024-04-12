import axios from "axios";
//import { JSON_SERVER_URL } from '../utils/utils';

const sendModelData = async (
  utility,
  region,
  size,
  energy,
  state,
  county,
  year
) => {
  const requestOptions = {
    method: "post",
    headers: {
      "Content-Type": "application/json",
    },
    url: `/api/model`,
    data: {
      utility,
      region,
      size,
      energy,
      state,
      county,
      year,
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

export default sendModelData;
