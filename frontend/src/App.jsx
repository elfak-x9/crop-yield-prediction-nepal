import { Routes, Route } from "react-router-dom";
import Navbar from "./components/layout/navbar";
import Home from "./pages/home";
import Prediction from "./pages/prediction";
import Dashboard from "./pages/dashboard";
import About from "./pages/about";
import NotFound from "./pages/notfound";

function App() {
  return (
    <>
      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
export default App;
