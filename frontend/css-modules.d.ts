declare module '*.module.css' {
  const classes: { readonly [key: string]: string }
  export default classes
}

declare module '*.module.scss' {
  const classes: { readonly [key: string]: string }
  export default classes
}

declare module '*.module.sass' {
  const classes: { readonly [key: string]: string }
  export default classes
}

// Specific declaration for page.module.css
declare module './page.module.css' {
  const classes: {
    readonly main: string
    readonly container: string
    readonly header: string
    readonly title: string
    readonly subtitle: string
    readonly content: string
    readonly section: string
    readonly inputGroup: string
    readonly label: string
    readonly textarea: string
    readonly button: string
    readonly outputGroup: string
    readonly output: string
    readonly toggleSection: string
    readonly toggleButton: string
    readonly uploadGroup: string
    readonly fileInput: string
    readonly docxResults: string
    readonly resultsTitle: string
    readonly resultItem: string
    readonly resultOriginal: string
    readonly resultHumanized: string
    readonly qualityScore: string
    readonly error: string
  }
  export default classes
}
