import axios from "axios";

export const predictYield = async (data) => {
  const response = await axios.post(
    "http://127.0.0.1:8000/api/predict/",
    data
  );

  return response.data;
};