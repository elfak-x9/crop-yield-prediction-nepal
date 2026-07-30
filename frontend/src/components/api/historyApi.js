import axios from "axios";

const API_URL = "http://127.0.0.1:8000/api/history/";

// Get prediction history
export const getHistory = async () => {
  const response = await axios.get(API_URL);
  return response.data;
};

// Delete prediction
export const deleteHistory = async (id) => {
  const response = await axios.delete(`${API_URL}${id}/`);
  return response.data;
};