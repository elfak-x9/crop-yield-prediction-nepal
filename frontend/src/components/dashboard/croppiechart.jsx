import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";

import { Pie } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

function CropPieChart({ history }) {
  const cropCount = {};

  history.forEach((item) => {
    cropCount[item.crop] = (cropCount[item.crop] || 0) + 1;
  });

  const data = {
    labels: Object.keys(cropCount),
    datasets: [
      {
        data: Object.values(cropCount),
        backgroundColor: [
          "#22c55e",
          "#3b82f6",
          "#f59e0b",
          "#ef4444",
          "#8b5cf6",
          "#14b8a6",
        ],
      },
    ],
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
      <h2 className="text-2xl font-bold text-center mb-4">
        Crop Distribution
      </h2>

      <div className="h-80">
        <Pie
          data={data}
          options={{
            maintainAspectRatio: false,
          }}
        />
      </div>
    </div>
  );
}

export default CropPieChart;