/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          950: "#0b0d10",
          900: "#111417",
          800: "#181c20",
          700: "#20252b",
          600: "#2a3038",
          500: "#3a4149",
        },
        accent: {
          500: "#4f9cff",
          600: "#3b82f6",
        },
      },
    },
  },
  plugins: [],
};
