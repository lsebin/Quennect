import axios from 'axios';
import { JSON_SERVER_URL, SEARCH_PAGE_ITEMS_PER_PAGE } from '../utils/utils';

const searchItem = async (
  searchQuery = '',
  page = 1,
  limit = SEARCH_PAGE_ITEMS_PER_PAGE,
  sortBy = 'urgent',
  priceRangeLow = 0,
  priceRangeHigh = 99999,
  category = 'all',
) => {
  const requestOptions = {
    method: 'get',
    headers: {
      'Content-Type': 'application/json',
    },
    url: `${JSON_SERVER_URL}/posts/search?searchQuery=${searchQuery}&page=${page}&limit=${limit}&sortBy=${sortBy}&priceRangeLow=${priceRangeLow}&priceRangeHigh=${priceRangeHigh}&category=${category === 'all' ? '.*' : category}`,
  };

  try {
    const response = await axios(requestOptions);
    return response;
  } catch (err) {
    // console.error(err.message);
    return err.response;
  }
};

export default searchItem;
