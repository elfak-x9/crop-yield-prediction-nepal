import { useEffect, useState } from "react";
import { getHistory, deleteHistory } from "../components/api/historyApi";
import CropBarChart from "../components/dashboard/cropbarchart";
import CropPieChart from "../components/dashboard/croppiechart";

function Dashboard() {
  const [history, setHistory] = useState([]);
  const [search, setSearch] = useState("");
  const [locationFilter, setLocationFilter] = useState("All");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getHistory();
        console.log(data);
        setHistory(data);
      } catch (error) {
        console.error("Error fetching history:", error);
      }
    };

    fetchData();
  }, []);

  // Statistics
  const totalPredictions = history.length;

  const totalLandArea = history.reduce(
    (sum, item) => sum + Number(item.land_area),
    0
  );

  const averageYield =
    history.length > 0
      ? (
          history.reduce(
            (sum, item) => sum + Number(item.predicted_yield),
            0
          ) / history.length
        ).toFixed(2)
      : 0;

  // Unique Locations
  const locations = [
    "All",
    ...new Set(history.map((item) => item.location)),
  ];

  // Search + Filter
  const filteredHistory = history.filter((item) => {
    const matchCrop = item.crop
      .toLowerCase()
      .includes(search.toLowerCase());

    const matchLocation =
      locationFilter === "All" ||
      item.location === locationFilter;

    return matchCrop && matchLocation;
  });

  // Export CSV
  const exportCSV = () => {
    const header = [
      "Crop",
      "Location",
      "Land Area",
      "Predicted Yield",
    ];

    const rows = filteredHistory.map((item) => [
      item.crop,
      item.location,
      item.land_area,
      item.predicted_yield,
    ]);

    const csv = [header, ...rows]
      .map((e) => e.join(","))
      .join("\n");

    const blob = new Blob([csv], {
      type: "text/csv",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "prediction_history.csv";
    link.click();

    URL.revokeObjectURL(url);
  };

  // Delete Prediction
  const handleDelete = async (id) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this prediction?"
    );

    if (!confirmDelete) return;

    try {
      await deleteHistory(id);

      setHistory((prevHistory) =>
        prevHistory.filter((item) => item.id !== id)
      );

      alert("Prediction deleted successfully.");
    } catch (error) {
      console.error(error);
      alert("Failed to delete prediction.");
    }
  };

  return (
    <div className="min-h-screen bg-green-50 pt-24 px-6 pb-6">
      <h1 className="text-4xl font-bold text-green-700 mb-6">
        Prediction Dashboard
      </h1>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-xl shadow p-4 text-center">
          <h2 className="text-gray-500 font-semibold">
            Total Predictions
          </h2>
          <p className="text-3xl font-bold text-green-700 mt-2">
            {totalPredictions}
          </p>
        </div>

        <div className="bg-white rounded-xl shadow p-4 text-center">
          <h2 className="text-gray-500 font-semibold">
            Total Land Area
          </h2>
          <p className="text-3xl font-bold text-blue-600 mt-2">
            {totalLandArea} Ha
          </p>
        </div>

        <div className="bg-white rounded-xl shadow p-4 text-center">
          <h2 className="text-gray-500 font-semibold">
            Average Yield
          </h2>
          <p className="text-3xl font-bold text-orange-500 mt-2">
            {averageYield}
          </p>
        </div>
      </div>

      {/* Search + Filter */}
      <div className="flex flex-col md:flex-row gap-4 justify-between mb-6">
        <input
          type="text"
          placeholder="🔍 Search Crop..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="border rounded-lg px-4 py-2 w-full md:w-72"
        />

        <select
          value={locationFilter}
          onChange={(e) => setLocationFilter(e.target.value)}
          className="border rounded-lg px-4 py-2 w-full md:w-60"
        >
          {locations.map((loc) => (
            <option key={loc} value={loc}>
              {loc}
            </option>
          ))}
        </select>

        <button
          onClick={exportCSV}
          className="bg-green-600 text-white px-5 py-2 rounded-lg hover:bg-green-700"
        >
          Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto bg-white rounded-xl shadow">
        <table className="w-full">
          <thead className="bg-green-600 text-white">
            <tr>
              <th className="py-3">Crop</th>
              <th>Location</th>
              <th>Land Area</th>
              <th>Predicted Yield</th>
              <th>Action</th>
            </tr>
          </thead>

          <tbody>
            {filteredHistory.length > 0 ? (
              filteredHistory.map((item) => (
                <tr
                  key={item.id}
                  className="text-center border-b hover:bg-gray-100"
                >
                  <td className="py-3">{item.crop}</td>
                  <td>{item.location}</td>
                  <td>{item.land_area}</td>
                  <td>{item.predicted_yield}</td>
                  <td>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="5" className="py-6 text-center text-gray-500">
                  No prediction history found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <CropBarChart history={filteredHistory} />
        <CropPieChart history={filteredHistory} />
      </div>
    </div>
  );
}

export default Dashboard;