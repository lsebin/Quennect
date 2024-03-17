import { HOME_ITEM_PER_PAGE } from '../utils/utils';
import searchItem from './search';

export const displayRecent = async (
  page = 1,
  limit = HOME_ITEM_PER_PAGE,
) => {
  // try {
  const response = await searchItem('.*', page, limit, 'recent');
  return response;
  // } catch (err) {
  //   return err.response;
  // }
};

export const displayUrgent = async (
  page = 1,
  limit = HOME_ITEM_PER_PAGE,
) => {
  // try {
  const response = await searchItem('.*', page, limit, 'urgent');
  return response;
  // } catch (err) {
  //   return err.response;
  // }
};
