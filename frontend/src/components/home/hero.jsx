import heroImage from "../../assets/image/hero.jpg";


function Hero() {
  return (
    <section
      className="min-h-screen bg-cover bg-center flex items-center"
      style={{
        backgroundImage: `linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), url(${heroImage})`,
      }}
    >
      <div className="max-w-7xl mx-auto px-6">
        <div className="max-w-2xl">

          <h1 className="text-6xl font-bold text-white leading-tight">
            Crop Yield Prediction
          </h1>

          <p className="mt-6 text-xl text-gray-200">
            Predict crop yield using AI and agricultural data.
            Help farmers make smarter decisions with machine learning.
          </p>

          <div className="mt-10 flex gap-5">
            <button className="bg-green-600 px-7 py-3 rounded-lg hover:bg-green-700">
              Start Prediction
            </button>

            <button className="border border-white px-7 py-3 rounded-lg hover:bg-white hover:text-black transition">
              Learn More
            </button>
          </div>

        </div>
      </div>
    </section>
  );
}

export default Hero;