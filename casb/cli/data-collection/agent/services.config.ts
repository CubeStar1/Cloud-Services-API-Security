import { CREDENTIALS } from "./credentials.config.js";

const googleSignInActions = [
  "Click 'Sign in with Google' or 'Continue with Google'",
  `Type '${CREDENTIALS.google.email}' into the email input`,
  "Click 'Next'",
  `Type '${CREDENTIALS.google.password}' into the password input`,
  "Click 'Next' or 'Sign in'",
  "Wait for 2 seconds"
];

export const CLOUD_SERVICES = [
  {
    name: 'google-workspace',
    url: 'https://workspace.google.com',
    actions: [
      "Click the get started button ",
      "Fill out the form with the necessary information",
      "Make sure to complete the onboarding flow"
    ]
  },
  {
    name: 'google-cloud',
    url: 'https://cloud.google.com',
    actions: [
      "Click 'Get started for free'",
     "Click 'Sign in with Google'",
      ...googleSignInActions,
    ]
  },
  {
    name: 'github',
    url: 'https://github.com',
    actions: [
      "Click 'Sign up'",
      "Create a new account"
    ]
  },
  {
    name: 'slack',
    url: 'https://slack.com/get-started',
    actions: [
      "Click 'Sign in with Google'",
      ...googleSignInActions,
      "Click 'Create a new workspace'"
    ]
  },
  {
    name: 'trello',
    url: 'https://trello.com/signup',
    actions: [
      "Click 'Continue with Google'",
      ...googleSignInActions,
      "Click 'Create account' after sign-in"
    ]
  },
  {
    name: 'asana',
    url: 'https://app.asana.com/-/login',
    actions: [
      "Click 'Continue with Google'",
      ...googleSignInActions,
      "Click 'Continue' after Google sign-in"
    ]
  },
  {
    name: 'zoom',
    url: 'https://zoom.us',
    actions: [
      "Click 'Sign Up, It's Free'",
      "Click 'Sign Up for Free'"
    ]
  },
  {
    name: 'microsoft-teams',
    url: 'https://www.microsoft.com/microsoft-teams',
    actions: [
      "Click 'Sign up for free'",
      "Click 'Get Teams for free'"
    ]
  },
  {
    name: 'dropbox',
    url: 'https://www.dropbox.com/login',
    actions: [
      "Click 'Continue with Google'",
      ...googleSignInActions,
      "Click 'Agree and create account' after sign-in"
    ]
  },
  {
    name: 'notion',
    url: 'https://www.notion.so/signup',
    actions: [
      "Click 'Continue with Google'",
      ...googleSignInActions,
      "Click 'Continue' after Google sign-in"
    ]
  },
  {
    name: 'airtable',
    url: 'https://airtable.com/signup',
    actions: [
      "Click 'Continue with Google'",
      ...googleSignInActions,
      "Click 'Continue' after sign-in"
    ]
  },
  {
    name: 'calendly',
    url: 'https://calendly.com/signup',
    actions: [
      "Click 'Sign up with Google'",
      ...googleSignInActions,
      "Click 'Continue' after Google sign-in"
    ]
  },
  {
    name: 'figma',
    url: 'https://www.figma.com/signup',
    actions: [
      "Click 'Continue with Google'",
      ...googleSignInActions,
      "Click 'Continue' after sign-in"
    ]
  },
  {
    name: 'confluence',
    url: 'https://www.atlassian.com/software/confluence',
    actions: [
      "Click 'Get it free'",
      "Click 'Try free'"
    ]
  },
  {
    name: 'cisco-webex',
    url: 'https://www.webex.com',
    actions: [
      "Click 'Sign up'",
      "Click 'Start for free'"
    ]
  },
  {
    name: 'gitlab',
    url: 'https://gitlab.com/users/sign_in',
    actions: [
      "Click 'Google'",
      ...googleSignInActions,
      "Click 'Authorize' after Google sign-in"
    ]
  },
  {
    name: 'heroku',
    url: 'https://signup.heroku.com',
    actions: [
      "Click 'Sign up with Google'",
      ...googleSignInActions,
      "Click 'Create account' after sign-in"
    ]
  },
  {
    name: 'firebase',
    url: 'https://firebase.google.com',
    actions: [
      "Click 'Get started'",
      "Click 'Try for free'"
    ]
  },
  {
    name: 'cloudflare',
    url: 'https://www.cloudflare.com',
    actions: [
      "Click 'Sign up'",
      "Click 'Create a free account'"
    ]
  }
]; 