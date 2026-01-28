import dotenv from "dotenv";
dotenv.config();

export const CREDENTIALS = {
  google: {
    email: process.env.GOOGLE_EMAIL,
    password: process.env.GOOGLE_PASSWORD
  }
}; 