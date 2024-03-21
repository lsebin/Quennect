import axios from "axios";
//import { JSON_SERVER_URL, SEARCH_PAGE_ITEMS_PER_PAGE } from "../utils/utils";
import { JSON_SERVER_URL, API_KEY } from "../utils/utils";

const submit = async (inputText) => {
  /*
  const requestOptions = {
    method: "post",
    headers: {
      "Content-Type": "application/json",
      Authorization: API_KEY,
    },
    url: `${JSON_SERVER_URL}/posts/search?searchQuery=${searchQuery}&page=${page}&limit=${limit}&sortBy=${sortBy}&priceRangeLow=${priceRangeLow}&priceRangeHigh=${priceRangeHigh}&category=${
      category === "all" ? ".*" : category
    }`,
  };
  */

  try {
    //const response = await axios(requestOptions);
    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        // Request data
        model: "gpt-3.5-turbo", // Use GPT-3.5 Turbo model
        messages: [
          {
            role: "user",
            content: inputText,
          },
        ],
        //max_tokens: 50, // Adjust as needed
      },
      {
        // Request options
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + API_KEY,
        },
        //timeout: 5000, // 5 seconds timeout
        //responseType: "json",
      }
    );
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export default submit;
