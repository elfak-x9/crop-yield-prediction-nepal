import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="fixed top-0 left-0 w-full z-50 bg-green-900/80 backdrop-blur-md shadow-lg">
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">

        {/* Logo */}
        <div className="flex items-center gap-2">
          <span className="text-3xl">🌱</span>
          <h1 className="text-2xl font-bold text-green-700">
            Crop Yield Prediction
          </h1>
        </div>

        {/* Navigation Links */}
        <div className="flex gap-8 text-lg font-medium">
          <Link
            to="/"
            className="hover:text-green-600 transition duration-300"
          >
            Home
          </Link>

          <Link
            to="/prediction"
            className="hover:text-green-600 transition duration-300"
          >
            Prediction
          </Link>

          <Link
            to="/statistics"
            className="hover:text-green-600 transition duration-300"
          >
            Statistics
          </Link>

          <Link
            to="/about"
            className="hover:text-green-600 transition duration-300"
          >
            About
          </Link>
        </div>

        {/* Button */}
        <button className="bg-green-600 text-white px-5 py-2 rounded-lg hover:bg-green-700 transition">
          Get Started
        </button>

      </div>
    </nav>
  );
}

export default Navbar;