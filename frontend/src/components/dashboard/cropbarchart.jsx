import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

import { Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function CropBarChart({ history }) {
  const data = {
    labels: history.map((item) => item.crop),
    datasets: [
      {
        label: "Predicted Yield",
        data: history.map((item) => Number(item.predicted_yield)),
        backgroundColor: "#22c55e",
        borderColor: "#16a34a",
        borderWidth: 2,

        // Prevent the bar from stretching
        barThickness: 60,
        maxBarThickness: 60,
        categoryPercentage: 0.4,
        barPercentage: 0.5,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,

    plugins: {
      legend: {
        position: "top",
      },
    },

    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="bg-black rounded-xl p-6 mt-8 shadow-lg">
      <h2 className="text-3xl font-bold text-center text-white mb-6">
        Predicted Yield by Crop
      </h2>

      <div style={{ height: "220px", width: "100%" }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
}

export default CropBarChart;